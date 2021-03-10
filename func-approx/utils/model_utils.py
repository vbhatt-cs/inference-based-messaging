from ray.rllib.utils import try_import_tf

tf = try_import_tf()


def build_cnn(cnn_in, filters, activation, name="conv"):
    """
    Helper to build CNN
    Args:
        cnn_in: Input tensor to CNN
        filters: List of [channels, kernel, stride]
        activation: Activation function
        name: Name of the layers

    Returns:
        Output tensor of the convolution
    """
    last_layer = cnn_in
    for i, (out_size, kernel, stride) in enumerate(filters, 1):
        if not isinstance(kernel, (tuple, list)):
            kernel = (kernel, kernel)
        if not isinstance(stride, (tuple, list)):
            stride = (stride, stride)

        last_layer = tf.keras.layers.Conv2D(
            out_size,
            kernel,
            strides=stride,
            activation=activation,
            padding="same",
            kernel_initializer=tf.initializers.variance_scaling(),
            name=f"{name}_{i}",
        )(last_layer)

    return last_layer


def build_fc(fc_in, hiddens, activation, name="fc", return_vars=False):
    """
    Helper to build FC
    Args:
        fc_in: Input tensor to FC
        hiddens: List of number of hidden neurons
        activation: Activation function
        name: Name of the layers
        return_vars: True if model variables should be returned

    Returns:
        Output tensor of the MLP
    """
    last_layer = fc_in
    var_list = []
    for i, size in enumerate(hiddens, 1):
        fc_layer = tf.keras.layers.Dense(
            size,
            name=f"{name}_{i}",
            activation=activation,
            kernel_initializer=tf.initializers.variance_scaling(),
        )
        last_layer = fc_layer(last_layer)
        var_list.extend(fc_layer.variables)
    fc_out = last_layer
    if return_vars:
        return fc_out, var_list
    else:
        return fc_out


def build_lstm(
    in_tensors,
    state_in_h,
    state_in_c,
    seq_in,
    cell_size,
    add_cpc=False,
    cpc_params=None,
    name="lstm",
):
    """
    Helper to build LSTM with optional Contrastive Predictive Coding loss
    Args:
        in_tensors: List of LSTM inputs (without time axis)
        state_in_h: Input state to LSTM
        state_in_c: Input state to LSTM
        seq_in: Sequence lengths
        cell_size: LSTM cell size
        add_cpc: True if CPC loss needs to be added
        cpc_params: Dict for CPC parms
            "cpc_len": Length of predictions
            "name": Name of the CPC layers
        name: Name of the layers

    Returns:
        List [output tensor of LSTM, other outputs]

        If add_cpc_loss is True, other outputs is a list with
            [output state h, output state c, cpc loss tensor]
        Else, it is [output state h, output state c]
    """
    T = tf.reduce_max(seq_in)
    lstm_ins = []
    for x in in_tensors:
        x_shape = x.shape.as_list()
        if len(x_shape) == 2:
            x_f = x_shape[-1]
        else:
            x_f = 1
        x_r = tf.reshape(x, [-1, T, x_f])
        lstm_ins.append(x_r)

    if len(lstm_ins) > 1:
        lstm_in = tf.keras.layers.Concatenate(name=f"{name}_in")(lstm_ins)
    else:
        lstm_in = lstm_ins[0]
    lstm_out, state_h, state_c = tf.keras.layers.LSTM(
        cell_size,
        kernel_initializer=tf.initializers.variance_scaling(),
        recurrent_initializer=tf.initializers.variance_scaling(),
        recurrent_activation="sigmoid",
        return_sequences=True,
        return_state=True,
        name=name,
    )(
        inputs=lstm_in,
        mask=tf.sequence_mask(seq_in),
        initial_state=[state_in_h, state_in_c],
    )

    if add_cpc:
        if cpc_params is None:
            cpc_params = {}
        cpc_len = cpc_params.get("cpc_len", 20)
        cpc_name = cpc_params.get("name", "cpc")
        cpc_code_size = cpc_params.get("cpc_code_size", 64)

        cpc_ins, cpc_preds = build_cpc(
            lstm_in, lstm_out, cpc_len, cpc_code_size, name=cpc_name,
        )
        return lstm_out, [state_h, state_c, cpc_ins, cpc_preds]
    else:
        return lstm_out, [state_h, state_c]


def build_cpc(lstm_in, lstm_out, length, code_size, name="cpc"):
    """
    Creates the encoding layers for Contrastive Predictive Coding loss
    Args:
        lstm_in: Input to the LSTM (input before encoding)
        lstm_out: Output of the LSTM (output before encoding)
        length: CPC prediction length
        code_size: CPC encoding space
        name: Name of the layers

    Returns:
        Input and output encodings for CPC
    """
    lstm_in_enc = tf.keras.layers.Dense(
        code_size,
        activation=tf.keras.activations.linear,
        kernel_initializer=tf.initializers.variance_scaling(),
        name=f"{name}_input_enc",
    )(
        lstm_in
    )  # Shape [B, T, code_size]

    outs = []
    for k in range(1, length + 1):
        lstm_out_enc = tf.keras.layers.Dense(
            code_size,
            activation=tf.keras.activations.linear,
            kernel_initializer=tf.initializers.variance_scaling(),
            name=f"{name}_output_enc_{k}",
        )(lstm_out)
        outs.append(lstm_out_enc)

    cpc_preds = tf.stack(outs, axis=2)  # Shape [B, T, length, code_size]
    return lstm_in_enc, cpc_preds


class CPCLayer(tf.keras.layers.Layer):
    """
    Layer to calculate the Contrastive Predictive Coding loss given the input and output
    encodings
    """

    def __init__(self, k, **kwargs):
        """
        Args:
            k: Timesteps in future to predict
            **kwargs: Layer args
        """
        super().__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        """
        Args:
            inputs: List of [encoded_inputs, encoded_outputs]

        Returns:
            CPC loss tensor
        """
        lstm_in_enc = inputs[0]
        lstm_out_enc = inputs[1]
        shape = tf.shape(lstm_in_enc)
        T = shape[0]
        B = shape[1]
        labels = tf.reshape(tf.tile(tf.range(B), multiples=[T - self.k]), [T - self.k, B])
        in_to_dot = lstm_in_enc[self.k :]
        out_to_dot = lstm_out_enc[: -self.k]
        cpc_logits = tf.matmul(
            out_to_dot, in_to_dot, transpose_b=True, name="cpc_logits"
        )  # Shape [T-k, B, B]
        return tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=cpc_logits, labels=labels
            ),
            axis=-1,
        )  # Shape [T-k]

    def get_config(self):
        config = {"k": self.k}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
