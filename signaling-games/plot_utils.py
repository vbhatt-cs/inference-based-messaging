from matplotlib import pyplot as plt
from matplotlib_venn import venn3


def plot_confusion_matrix(cm, cmap=plt.cm.Blues):
    """
    Args:
        cm (np.ndarray): Confusion matrix to plot
        cmap: Color map to be used in matplotlib's imshow

    Returns:
        Figure and axis on which the confusion matrix is plotted
    """
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Receiver's action", fontsize=14)
    ax.set_ylabel("Sender's state", fontsize=14)
    # Loop over data dimensions and create text annotations.
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig, ax


def plot_venn(ns):
    n_diff, n01, n12, n02, n012 = ns
    fig, ax = plt.subplots()
    venn3(
        subsets=(
            n_diff + n12 - n012,
            n_diff + n02 - n012,
            n01 - n012,
            n_diff + n01 - n012,
            n02 - n012,
            n12 - n012,
            n012,
        ),
        set_labels=(r"$s_1$", r"$s_2$", r"$s_3$"),
        ax=ax,
    )

    return fig, ax
