import argparse
import os

import numpy as np
import pandas as pd


def save_exp(exp_dir):
    """
    Stores the rewards and corresponding time-steps for each run (since other parts of the
    logs are not used in the final table). Also calculates and store the mean and standard
    error over all the repetitions. Change the `key` variable if something other than the
    reward needs to be saved.
    Args:
        exp_dir: The directory where the logs are stored (each repetition will have a
        directory inside this directory with `progress.csv` file in it)
    """
    key = "policy_reward_mean/receiver"
    steps_key = "timesteps_total"
    all_scalars = []

    for d in os.listdir(exp_dir):
        if os.path.isdir(f"{exp_dir}/{d}"):
            res_file = f"{exp_dir}/{d}/progress.csv"
            res_df = pd.read_csv(res_file)
            scalars = res_df[key].to_numpy()
            steps = res_df[steps_key].to_numpy()
            np.save(f"{exp_dir}/{d}/_scalars", scalars)
            np.save(f"{exp_dir}/{d}/_steps", steps)
            all_scalars.append(scalars)

    res_df = pd.DataFrame(all_scalars)
    mean = res_df.mean(axis=0)
    sem = res_df.sem(axis=0)

    np.save(f"{exp_dir}_mean", mean)
    np.save(f"{exp_dir}_sem", sem)


def split_runs(logdir, max_steps=3e8, avg_steps=5e5, good_threshold=13):
    """
    Split the runs into good runs and bad runs based on the obtained final rewards.
    Args:
        logdir: The base directory of logs (contains directories for each config)
        max_steps: Time-steps at which the final rewards should be calculated
        avg_steps: Number of steps before `max_steps` which should be included for
        calculating the average final reward
        good_threshold: The reward threshold above which a run is considered good

    Returns:

    """
    for d in os.listdir(logdir):
        scalars = []
        good_scalars = []
        exp_dir = f"{logdir}/{d}/impala_cpc_sa/"
        for d2 in os.listdir(exp_dir):
            if os.path.isdir(f"{exp_dir}/{d2}"):
                scalar = np.load(f"{exp_dir}/{d2}/_scalars.npy")
                x = np.load(f"{exp_dir}/{d2}/_steps.npy")
                idx_end = np.argmax(x > max_steps) - 1
                idx_begin = np.argmax(x > max_steps - avg_steps)
                scalar = np.mean(scalar[idx_begin:idx_end])
                scalars.append(scalar)
                if scalar >= good_threshold:
                    good_scalars.append(scalar)

        from scipy.stats import stats
        print(f"Experiment: {d}")
        print(f"Good: Mean: {np.mean(good_scalars)}, Sem: {stats.sem(good_scalars)}")
        print(f"All: Mean: {np.mean(scalars)}, Sem: {stats.sem(scalars)}")


def main(analyze):
    logdir = "logs"
    if analyze:
        split_runs(logdir)

    else:
        for d in os.listdir(logdir):
            exp_dir = f"{logdir}/{d}/impala_cpc_sa/"
            save_exp(exp_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--analyze", help="set for analysis", action="store_true", default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.analyze)
