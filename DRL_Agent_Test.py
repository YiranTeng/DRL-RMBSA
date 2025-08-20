
import pickle

import time
import numpy as np
from scipy.io import loadmat
from stable_baselines3.common.monitor import Monitor
from ppo_mask_new import MaskablePPO
import env

import gym
from multiprocessing import Process


def evaluate_DRL_Agent(
    env: "OpticalNetworkEnv",
    DRL_Agent,
    n_eval_episodes=10,
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
):
    episode_rewards, episode_lengths, episode_bp, episode_brbp = [], [], [], []
    print('gogo')
    inference_time = 0
    for _ in range(n_eval_episodes):
        print(_)
        observation = env.reset()
        done, _ = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            t1 = time.time()
            action = DRL_Agent.predict(observation, action_masks=env.action_masks(), deterministic=True)
            t2 = time.time()
            inference_time += t2 - t1
            observation, reward, done, _ = env.step(action[0])
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()

        episode_bp1 = (env.episode_services_processed - env.episode_services_accepted) / (env.episode_services_processed)
        episode_brbp1 = (env.episode_bit_rate_requested - env.episode_bit_rate_provisioned) / (env.episode_bit_rate_requested)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_bp.append(episode_bp1)
        episode_brbp.append(episode_brbp1)


    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_bp = np.mean(episode_bp)
    mean_brbp = np.mean(episode_brbp)
    print('inference_time:' + str(inference_time))

    return mean_reward, std_reward, mean_bp, mean_brbp

def DRL_RBMSA(traffic_load):
    env_args = dict(topology=topology,
                    seed=10,
                    mean_service_holding_time=traffic_load,
                    mean_service_inter_arrival_time=1,
                    bands=bands,
                    ksp_node_lists=ksp_node_lists,
                    modulation_table=modulation_table,
                    GSNR_table=GSNR_table,
                    GSNR_record_episode=number_episodes*0,
                    use_mask=True,
                    episode_length=1000)
    log_dir = "./BP/Test/DRL_RMBSA/" + str(traffic_load)
    env = gym.make("DRL-RMBSA-v0", **env_args)
    env_sap_ff = Monitor(env, log_dir, info_keywords=monitor_info_keywords)

    model_path = "best_models/best_model_DRL_RMBSA.zip"
    DRL_Agent = MaskablePPO.load(model_path, device='cpu')

    # run the heuristic for a number of episodes
    mean_reward_sap_ff, std_reward_sap_ff, mean_bp, mean_brbp = evaluate_DRL_Agent(env_sap_ff, DRL_Agent, n_eval_episodes=number_episodes)
    mean_lightpath_GSNR, band_usage, mf_usage, path_usage = env_sap_ff.get_lightpath_mean_GSNR()

    # plot_links_channels_state(env_sap_ff.channels_state)

    print('DRL_Agent ' + 'load: ' + str(traffic_load))
    print('service blocking:', mean_bp)
    print('bit_rate blocking:', mean_brbp)
    print('lightpath GSNR:', mean_lightpath_GSNR)
    print('band_usage:', np.round(band_usage,3)*100)
    print('mf_usage:', np.round(mf_usage,3)*100)
    print('path_usage:', np.round(path_usage,3)*100)

def DRL_RBMSA_PCA(traffic_load):
    env_args = dict(topology=topology,
                    seed=10,
                    mean_service_holding_time=traffic_load,
                    mean_service_inter_arrival_time=1,
                    bands=bands,
                    ksp_node_lists=ksp_node_lists,
                    modulation_table=modulation_table,
                    GSNR_table=GSNR_table,
                    GSNR_record_episode=number_episodes*0,
                    use_mask=True,
                    episode_length=1000)
    log_dir = "./BP/Test/DRL_RMBSA_PCA/" + str(traffic_load)
    env = gym.make("DRL-RMBSA-PCA-v0", **env_args)
    env_sap_ff = Monitor(env, log_dir, info_keywords=monitor_info_keywords)

    model_path = "best_models/best_model_DRL_RMBSA_PCA.zip"
    DRL_Agent = MaskablePPO.load(model_path, device='cpu')

    # run the heuristic for a number of episodes
    mean_reward_sap_ff, std_reward_sap_ff, mean_bp, mean_brbp = evaluate_DRL_Agent(env_sap_ff, DRL_Agent, n_eval_episodes=number_episodes)
    mean_lightpath_GSNR, band_usage, mf_usage, path_usage = env_sap_ff.get_lightpath_mean_GSNR()

    # plot_links_channels_state(env_sap_ff.channels_state)

    print('DRL_Agent ' + 'load: ' + str(traffic_load))
    print('service blocking:', mean_bp)
    print('bit_rate blocking:', mean_brbp)
    print('lightpath GSNR:', mean_lightpath_GSNR)
    print('band_usage:', np.round(band_usage,3)*100)
    print('mf_usage:', np.round(mf_usage,3)*100)
    print('path_usage:', np.round(path_usage,3)*100)


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

number_episodes = 20
traffic_load_list = [700,750,800,850,900,950]

if __name__ == '__main__':
    start = time.time()
    processes = []
    # test DRL_RMBSA
    for traffic_load in traffic_load_list:
        p = Process(target=DRL_RBMSA, args=(traffic_load,))
        p.start()
        processes.append(p)

    # test DRL_RMBSA_PCA
    # for traffic_load in traffic_load_list:
    #     p = Process(target=DRL_RBMSA_PCA, args=(traffic_load,))
    #     p.start()
    #     processes.append(p)

    processes.append(p)

    [p.join() for p in processes]

    end = time.time()
    print('run_time:' + str(end - start))