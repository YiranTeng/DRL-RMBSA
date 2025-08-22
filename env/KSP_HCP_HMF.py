from typing import Tuple, List

import gym
import numpy as np

from .mbeon_env import MBEON


class KSP_HCP_HMF(MBEON):
    def __init__(
        self,
        topology=None,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        bands: List = ['L', 'C', 'S'],
        node_request_probabilities=None,
        ksp_node_lists=None,
        modulation_table=None,
        GSNR_table=None,
        GSNR_record_episode=None,
        seed=None,
    ):
        super().__init__(
            topology=topology,
            episode_length=episode_length,
            load=mean_service_holding_time / mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            node_request_probabilities=node_request_probabilities,
            ksp_node_lists=ksp_node_lists,
            modulation_table=modulation_table,
            GSNR_table=GSNR_table,
            GSNR_record_episode=GSNR_record_episode,
            bands=bands,
            seed=seed,
            reset=False
        )

        assert modulation_table is not None and ksp_node_lists is not None, 'modulation_table and sd_pairs should be provided'
        assert len(ksp_node_lists) == 2

        self.num_bands = len(self.bands)

        self.generate_links_id()
        shape = 1

        self.observation_space = gym.spaces.Box(
            low=-100, high=100, dtype=np.float64, shape=(shape,)
        )

        self.action_space = gym.spaces.Discrete(self.k_paths * self.num_bands)

        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.reset(only_episode_counters = False)

    def observation(self):
        self.current_service.actions_info = self.preprocessing()
        return np.array([0])

    def reward(self):
       return 1

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)

    def generate_links_id(self):
        for node_pair, path_list in self.k_shortest_paths.items():
            for path in path_list:
                original_links = []
                for i in range(len(path.node_list) - 1):
                    original_links.append(self.topology[path.node_list[i]][path.node_list[i + 1]]['index'])
                path.links_id = np.array(original_links)

    def get_first_available_channels(self, all_channels_capacity: np.ndarray) -> int:
        available_channels_index = np.where(all_channels_capacity != 0)[0] # z
        available_channels_capacity = all_channels_capacity[available_channels_index] # b
        total_capacity = 0
        temp = [] # o
        for channel_id, channel_capacity in enumerate(available_channels_capacity):
            temp.append(channel_id)
            total_capacity += channel_capacity  # q
            if total_capacity >= self.current_service.bit_rate:
                channels = available_channels_index[temp]
                num_channels = channels.size
                channels_capacity = available_channels_capacity[temp]

                return channels, num_channels, channels_capacity

    def preprocessing(self):
        pc_map = np.full((self.k_paths,), dtype=np.float64, fill_value=0)
        for idp, path in enumerate(
                self.k_shortest_paths[self.current_service.source, self.current_service.destination]):
            total_channels_capacity_path = np.sum(
                np.concatenate([np.prod(self.channels_map[band_id][path.links_id, :], axis=0).flatten() *
                                path.channels_modulation_format[band_id] for band_id in range(self.num_bands)]))
            pc_map[idp] = total_channels_capacity_path

        while np.any(pc_map != 0):
            max_idp = np.where(pc_map == np.max(pc_map))[0]
            mf = []
            fre = []
            actions_information = []
            self.action_id = []
            for idp in max_idp:
                path = self.k_shortest_paths[self.current_service.source, self.current_service.destination][idp]
                for band_id in range(self.num_bands):
                    slots_state = np.prod(self.channels_map[band_id][path.links_id, :], axis=0).flatten()
                    all_channels_capacity = path.channels_modulation_format[band_id] * slots_state * self.basic_channel_capacity  # a
                    if np.sum(all_channels_capacity) < self.current_service.bit_rate:
                        continue
                    channels, num_required_channels, channels_capacity = self.get_first_available_channels(all_channels_capacity)
                    max_mf = np.max(channels_capacity) / 100
                    mf.append(max_mf)

                    max_f = np.max(80 * band_id + channels)
                    fre.append(max_f)
                    actions_information.append([path, band_id, channels])
                    self.action_id.append(idp * self.num_bands + band_id)

            if len(actions_information) > 0:
                mf = np.array(mf)
                fre = np.array(fre)
                assert mf.size == fre.size
                assert mf.size == len(actions_information)
                indices = np.where(mf == np.max(mf))[0]
                idx = np.where(fre[indices] == np.min(fre[indices]))[0][0]
                return actions_information[indices[idx]]
            else:
                pc_map[max_idp] = 0
        return None

def KSP_HCP_HMF_Action(env: KSP_HCP_HMF) -> int:
    return env.current_service.actions_info
