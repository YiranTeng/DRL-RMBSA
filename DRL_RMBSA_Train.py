
import pickle

import time
import numpy as np
from torch import nn
from scipy.io import loadmat
from Callbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from ppo_mask_new import MaskablePPO

import env


if __name__ == '__main__':
    topology_name = 'NSFNET'
    prefix = topology_name + '_'

    sd_pairs = loadmat(prefix + 'sd_pairs.mat')['sd_pairs']
    ksp_node_lists = loadmat(prefix + 'ksp_node_lists.mat')['node_lists'].tolist()
    modulation_table = loadmat(prefix + 'modulation_table.mat')['modulation_table']
    GSNR_table = loadmat(prefix + 'GSNR_table.mat')['GSNR_table']

    modulation_table = np.swapaxes(modulation_table, 1, 2)
    GSNR_table = np.swapaxes(GSNR_table, 1, 2)

    a = []
    for ele in ksp_node_lists:
        b = []
        for node_list in ele[0][0]:
            b.append(node_list.flatten())
        a.append(b)
    ksp_node_lists = [sd_pairs, a]

    k_paths = 5
    with open(f'{topology_name}_{k_paths}-paths_6-modulations.h5', 'rb') as f:
         topology = pickle.load(f)

    monitor_info_keywords=('episode_service_blocking_rate',
                           'episode_bit_rate_blocking_rate')

    monitor_args = dict(info_keywords=monitor_info_keywords)

    log_dir = "./BP/DRL_RMBSA"

    n_envs = 5
    episode_length = 1000
    total_timesteps = 80000000
    num_episodes = total_timesteps / (n_envs * episode_length)
    GSNR_record_episodes = num_episodes * 0.95
    training_start_timestep = 20000
    training_end_timestep = total_timesteps

    env_args = dict(topology=topology,
                    seed=10,  # the agent cannot proactively reject a request
                    mean_service_holding_time=900,
                    mean_service_inter_arrival_time=1,
                    bands=('L', 'C', 'S'),
                    ksp_node_lists=ksp_node_lists,
                    modulation_table=modulation_table,
                    GSNR_table=GSNR_table,
                    GSNR_record_episode=GSNR_record_episodes,
                    use_mask=True,
                    episode_length=episode_length)

    env = make_vec_env("DRL-RMBSA-v0",
                       monitor_dir=log_dir,
                       env_kwargs=env_args,
                       n_envs=n_envs,
                       monitor_kwargs=monitor_args,
                       vec_env_cls=SubprocVecEnv)
    #
    policy_args = dict(net_arch=[dict(pi=5*[128], vf=5*[128])], activation_fn=nn.ReLU)  # we use the elu activation function
    #
    agent = MaskablePPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_args, gamma=0.95, n_epochs=1, learning_rate=5e-5, n_steps=200, gae_lambda=1,
                batch_size=500, seed=10, device='cpu', training_start_timestep=training_start_timestep, training_end_timestep=training_end_timestep)

    callback = SaveOnBestTrainingRewardCallback(check_freq=50000, log_dir=log_dir, show_plot=False)

    #
    start = time.time()
    b = agent.learn(total_timesteps=total_timesteps, callback=callback)
    end = time.time()
    run_time = end - start






