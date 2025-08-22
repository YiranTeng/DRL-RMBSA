import pickle
from multiprocessing import Process

import numpy as np
import env
import time
from scipy.io import loadmat

import gym
from stable_baselines3.common.monitor import Monitor

from utils import evaluate_heuristic
from env.KSP_FB_FF import KSP_FB_FF_Action
from env.KSP_MinMaxF import KSP_MinMaxF_Action
from env.KSP_HCP_HMF import KSP_HCP_HMF_Action

# NSFNET

bands = ('L', 'C', 'S')
monitor_info_keywords = ('episode_service_blocking_rate',
                         'episode_bit_rate_blocking_rate')

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

def KSP_FB_FF(traffic_load):
    env_args = dict(topology=topology,
                    seed=10,  # the agent cannot proactively reject a request
                    mean_service_holding_time=traffic_load,
                    mean_service_inter_arrival_time=1,
                    bands=bands,
                    ksp_node_lists=ksp_node_lists,
                    modulation_table=modulation_table,
                    GSNR_table=GSNR_table,
                    GSNR_record_episode=number_episodes*0,
                    episode_length=1000)

    env = gym.make('KSP-FB-FF-v0', **env_args)
    algorithm = KSP_FB_FF_Action
    log_dir = "./BP/Test/KSP-FB-FF/" + str(traffic_load)

    env_sap_ff = Monitor(env, log_dir, info_keywords=monitor_info_keywords)

    # run the heuristic for a number of episodes
    mean_reward_sap_ff, std_reward_sap_ff, mean_bp, mean_brbp = evaluate_heuristic(env_sap_ff, algorithm, n_eval_episodes=number_episodes)
    mean_lightpath_GSNR, band_usage, mf_usage, path_usage = env_sap_ff.get_lightpath_mean_GSNR()

    # plot_links_channels_state(env_sap_ff.channels_state)

    print('KSP_FB_FF' + 'load: ' + str(traffic_load))
    print('service blocking:', mean_bp)
    print('bit_rate blocking:', mean_brbp)
    print('lightpath GSNR:', mean_lightpath_GSNR)
    print('band_usage:', np.round(band_usage,3)*100)
    print('mf_usage:', np.round(mf_usage,3)*100)
    print('path_usage:', np.round(path_usage,3)*100)

def KSP_MinMaxF(traffic_load):
    env_args = dict(topology=topology,
                    seed=10,  # the agent cannot proactively reject a request
                    mean_service_holding_time=traffic_load,
                    mean_service_inter_arrival_time=1,
                    bands=bands,
                    ksp_node_lists=ksp_node_lists,
                    modulation_table=modulation_table,
                    GSNR_table=GSNR_table,
                    GSNR_record_episode=number_episodes*0,
                    episode_length=1000)

    log_dir = "./BP/Test/KSP_MinMaxF/" + str(traffic_load)
    env = gym.make('KSP-MinMaxF-v0', **env_args)
    algorithm = KSP_MinMaxF_Action
    env_sap_ff = Monitor(env, log_dir, info_keywords=monitor_info_keywords)

    # run the heuristic for a number of episodes
    mean_reward_sap_ff, std_reward_sap_ff, mean_bp, mean_brbp = evaluate_heuristic(env_sap_ff, algorithm, n_eval_episodes=number_episodes)
    mean_lightpath_GSNR, band_usage, mf_usage, path_usage = env_sap_ff.get_lightpath_mean_GSNR()


    print('KSP_MinMaxF ' + 'load: ' + str(traffic_load))
    print('service blocking:', mean_bp)
    print('bit_rate blocking:', mean_brbp)
    print('lightpath GSNR:', mean_lightpath_GSNR)
    print('band_usage:', np.round(band_usage,3)*100)
    print('mf_usage:', np.round(mf_usage,3)*100)
    print('path_usage:', np.round(path_usage,3)*100)

def KSP_HCP_HMF(traffic_load):
    env_args = dict(topology=topology,
                    seed=10,  # the agent cannot proactively reject a request
                    mean_service_holding_time=traffic_load,
                    mean_service_inter_arrival_time=1,
                    bands=bands,
                    ksp_node_lists=ksp_node_lists,
                    modulation_table=modulation_table,
                    GSNR_table=GSNR_table,
                    GSNR_record_episode=number_episodes*0,
                    episode_length=1000)

    log_dir = "./BP/Test/KSP_HCP_HMF/" + str(traffic_load)
    env = gym.make('KSP-HCP-HMF-v0', **env_args)
    algorithm = KSP_HCP_HMF_Action
    env_sap_ff = Monitor(env, log_dir, info_keywords=monitor_info_keywords)

    # run the heuristic for a number of episodes
    mean_reward_sap_ff, std_reward_sap_ff, mean_bp, mean_brbp = evaluate_heuristic(env_sap_ff, algorithm, n_eval_episodes=number_episodes)
    mean_lightpath_GSNR, band_usage, mf_usage, path_usage = env_sap_ff.get_lightpath_mean_GSNR()

    # plot_links_channels_state(env_sap_ff.channels_state)

    print('KSP_HCP_HMF' + 'load: ' + str(traffic_load))
    print('service blocking:', mean_bp)
    print('bit_rate blocking:', mean_brbp)
    print('lightpath GSNR:', mean_lightpath_GSNR)
    print('band_usage:', np.round(band_usage,3)*100)
    print('mf_usage:', np.round(mf_usage,3)*100)
    print('path_usage:', np.round(path_usage,3)*100)

number_episodes = 200
traffic_load_list = [700,750,800,850,900,950]

if __name__ == '__main__':
    start = time.time()
    processes = []

    for traffic_load in traffic_load_list:
        p = Process(target=KSP_FB_FF, args=(traffic_load,))
        p.start()
        processes.append(p)

    # for traffic_load in traffic_load_list:
    #     p = Process(target=KSP_MinMaxF, args=(traffic_load,))
    #     p.start()
    #     processes.append(p)

    # for traffic_load in traffic_load_list:
    #     p = Process(target=KSP_HCP_HMF, args=(traffic_load,))
    #     p.start()
    #     processes.append(p)


    [p.join() for p in processes]

    end = time.time()
    print('run_time:' + str(end - start))