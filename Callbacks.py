import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from IPython.display import clear_output
from typing import Any, Dict, Generator, List, Optional, Union

import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter

X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"

# callback from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1, show_plot: bool=False):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.save_path1 = os.path.join(log_dir, 'best_model_bp')
        self.best_mean_reward = -np.inf
        self.best_mean_bp = 1
        self.show_plot = show_plot
        self.start = time.time()

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if self.show_plot and self.n_calls % self.check_freq == 0 and self.n_calls > 5001:
            plotting_average_window = 100

            training_data = pd.read_csv(self.log_dir + 'training.monitor.csv', skiprows=1)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))

            ax1.plot(np.convolve(training_data['r'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')

            ax2.semilogy(np.convolve(training_data['episode_service_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode service blocking rate')

            ax3.semilogy(np.convolve(training_data['episode_bit_rate_blocking_rate'], np.ones(plotting_average_window)/plotting_average_window, mode='valid'))

            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Episode bit rate blocking rate')

            # fig.get_size_inches()
            plt.tight_layout()
            plt.show()

        if self.n_calls % self.check_freq == 0:
            end = time.time()
            print('run time:' + str(end-self.start))
            self.start = time.time()
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            a, b = ts2xy1(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                mean_bp = np.mean(b[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if mean_bp < self.best_mean_bp:
                     self.best_mean_bp = mean_bp
                     # Example for saving best model
                     if self.verbose > 0:
                         print("Saving new best model to {}".format(self.save_path1))
                         self.model.save(self.save_path1)
                     self.record_point = self.n_calls / self.check_freq
                if self.verbose > 0:
                    clear_output(wait=True)
            print('best_bp_model saved in: ' + str(self.record_point))

        return True

def ts2xy1(data_frame: pd.DataFrame, x_axis: str):
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: the x and y output
    """
    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)
        y_var = data_frame.episode_service_blocking_rate.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.episode_service_blocking_rate.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = data_frame.episode_service_blocking_rate.values
    else:
        raise NotImplementedError
    return x_var, y_var