import argparse
import warnings

import ray
import tensorflow.compat.v1 as tf
from ray import tune
from ray.rllib.agents.registry import get_agent_class

from configs.gridworld_ma import configs as configs_ma
from configs.gridworld_sa import configs as configs_sa


def main(
    config,
    n_gpu,
    n_gpu_per_run,
    n_cpu,
    n_cpu_per_run,
    num_samples,
    seed,
    logdir,
    restore,
    log_graph,
):
    n_gpu_per_run = min(n_gpu_per_run, n_gpu)
    n_cpu_per_run = min(n_cpu_per_run, n_cpu)
    full_config = config.copy()
    # Below configs need to be changed before passing to RLlib
    del full_config["run"]
    del full_config["tot_steps"]
    del full_config["max_workers"]
    workers_left = n_cpu_per_run - 1
    full_config["num_gpus"] = n_gpu_per_run
    full_config["num_workers"] = min(workers_left, config["max_workers"])
    workers_left -= full_config["num_workers"]
    full_config["evaluation_num_workers"] = workers_left
    if num_samples == 1:
        full_config["seed"] = seed

    if log_graph:
        # Create an override to plot the tensorflow compute graph. Only used for
        # debugging and not during training
        if isinstance(config["run"], type):
            base = config["run"]
        else:
            base = get_agent_class(config["run"])

        class LogOverride(base):
            def _init(self, config, env_creator):
                super()._init(config, env_creator)
                try:
                    policy = self.get_policy(
                        list(config["multiagent"]["policies"].keys())[0]
                    )
                except IndexError:
                    policy = self.get_policy()
                self._file_writer = tf.summary.FileWriter(
                    logdir=self.logdir, graph=policy._sess.graph,
                )
                self._file_writer.flush()

        run = LogOverride
    else:
        run = config["run"]

    ray.init(num_cpus=n_cpu, num_gpus=n_gpu)
    tune.run(
        run,
        config=full_config,
        local_dir=logdir,
        stop={"timesteps_total": config["tot_steps"]},
        checkpoint_freq=int(config["tot_steps"] / 1e6),
        checkpoint_at_end=True,
        restore=restore,
        num_samples=num_samples,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_num", help="config number", type=int, default=0)
    parser.add_argument("--n_gpu", help="num GPUs to use ", type=int, default=1)
    parser.add_argument("--n_gpu_per_run", help="num GPUs per run", type=int, default=1)
    parser.add_argument("--n_cpu", help="num CPUs to use ", type=int, default=8)
    parser.add_argument("--n_cpu_per_run", help="num CPUs per run", type=int, default=8)
    parser.add_argument("--num_samples", help="repetitions", type=int, default=1)
    parser.add_argument("--seed", help="random seed", type=int, default=None)
    parser.add_argument("--logdir", help="logging base path", type=str, default="logs/")
    parser.add_argument(
        "--ma", help="set for multi-agent", action="store_true", default=False
    )
    parser.add_argument("--restore", help="checkpoint path", type=str, default=None)
    parser.add_argument(
        "--log_graph", help="log session graph", action="store_true", default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ma:
        try:
            config_num = args.config_num
            config = configs_ma[args.config_num]
        except IndexError:
            warnings.warn("Using default config")
            config = configs_ma[0]
            config_num = 0
        main(
            config=config,
            n_gpu=args.n_gpu,
            n_gpu_per_run=args.n_gpu_per_run,
            n_cpu=args.n_cpu,
            n_cpu_per_run=args.n_cpu_per_run,
            num_samples=args.num_samples,
            seed=args.seed,
            logdir=f"{args.logdir}/{config_num}",
            restore=args.restore,
            log_graph=args.log_graph,
        )
    else:
        main(
            config=configs_sa["impala"],
            n_gpu=args.n_gpu,
            n_gpu_per_run=args.n_gpu_per_run,
            n_cpu=args.n_cpu,
            n_cpu_per_run=args.n_cpu_per_run,
            num_samples=args.num_samples,
            seed=args.seed,
            logdir=args.logdir,
            restore=args.restore,
            log_graph=args.log_graph,
        )
