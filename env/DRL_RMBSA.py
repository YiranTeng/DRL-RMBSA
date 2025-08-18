from typing import Tuple, List
from utils import Path, Service

import gym
import numpy as np
import math
import copy
import itertools
import bisect
import time

from .mbeon_env import MBEON


class DRL_RMBSA(MBEON):
    def __init__(
        self,
        topology=None,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        bands: List = ['L', 'C', 'S'],
        use_mask=True,
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

        self.use_mask = 1 if use_mask else 0
        self.num_bands = len(self.bands)

        self.generate_adjacent_links()
        self.adjacent_links_list = self.generate_adjacent_links_list()
        self.generate_links_id()
        self.max_hops, self.path_vector_obs = self.generate_path_vector()

        shape = self.max_hops * self.k_paths + self.k_paths * self.num_bands * 5

        self.observation_space = gym.spaces.Box(
            low=-100, high=100, dtype=np.float64, shape=(shape,)
        )

        self.action_space = gym.spaces.Discrete(self.k_paths * self.num_bands + self.use_mask)

        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.reset(only_episode_counters = False)

    def step(self, action: int):
        assert action < self.action_space.n
        self.current_service.action_id = action
        if not self.current_service.valid_actions[action] or (self.use_mask and action == self.action_space.n - 1):
            return super().step(None)
        else:
            actions_info = self.current_service.actions_info[action]
            return super().step(actions_info)

    def observation(self):
        self.current_service.valid_actions, self.current_service.actions_info, spectrum_obs = self.preprocessing()
        links_obs = self.path_vector_obs[self.current_service.source, self.current_service.destination]
        return np.concatenate([links_obs, spectrum_obs.flatten()])

    def reward(self):
        if self.current_service.accepted:
            return 1
        else:
            return -1

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)

    def generate_adjacent_links(self):
        for key in self.k_shortest_paths:
            for path in self.k_shortest_paths[key]:
                adjacent_link_list = []
                link_index_list = []
                for idx, node in enumerate(path.node_list):
                    if idx + 1 != len(path.node_list):
                        mini_node_list = path.node_list[idx : idx+2]
                        link_index_list.append(self.topology[mini_node_list[0]][mini_node_list[1]]['index'])
                        for idx1, node1 in enumerate(mini_node_list):
                            for adjacent_node in self.topology[node1]:
                                if adjacent_node not in mini_node_list:
                                    adj_id = self.topology[node1][adjacent_node]['index']
                                    adjacent_link_list.append(adj_id)
                adjacent_link_list_copy = copy.deepcopy(adjacent_link_list)
                for node in adjacent_link_list_copy:
                    if node in link_index_list:
                        index = adjacent_link_list.index(node)
                        del adjacent_link_list[index]
                path.adjacent_link = adjacent_link_list

    def generate_links_id(self):
        for node_pair, path_list in self.k_shortest_paths.items():
            for path in path_list:
                original_links = []
                for i in range(len(path.node_list) - 1):
                    original_links.append(self.topology[path.node_list[i]][path.node_list[i + 1]]['index'])
                path.links_id = np.array(original_links)

    def generate_adjacent_links_list(self):
        adjacent_links_list = [[] for i in range(self.topology.number_of_edges())]
        for node in self.topology:
            temp = []
            for adjacent_node in self.topology[node]:
                link_id = self.topology[node][adjacent_node]['index']
                temp.append(link_id)
            for id, ele in enumerate(temp):
                temp1 = copy.deepcopy(temp)
                temp1.remove(ele)
                adjacent_links_list[ele].extend(temp1)
        adjacent_links_list= [list(dict.fromkeys(sublist)) for sublist in adjacent_links_list]

        return adjacent_links_list

    def generate_path_vector(self):
        path_hops_list = []
        for node_pair, path_list in self.k_shortest_paths.items():
            for path in path_list:
                path_hops_list.append(path.hops)
        max_hops = max(path_hops_list)

        path_vector_obs = {}
        for node_pair, path_list in self.k_shortest_paths.items():
            links_obs = np.full((self.k_paths, max_hops), fill_value=-1, dtype=np.float64)
            for idp, path in enumerate(path_list):
                links_obs[idp, :path.links_id.size] = path.links_id
            path_vector_obs[node_pair] = links_obs.flatten()

        return max_hops, path_vector_obs

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
        valid_actions = np.zeros([self.k_paths * self.num_bands + self.use_mask], dtype=bool)
        spectrum_obs = np.full((self.k_paths * self.num_bands, 5), fill_value=-1.0)
        actions_information = [None for i in range(self.action_space.n)]

        for idp, path in enumerate(
                self.k_shortest_paths[self.current_service.source, self.current_service.destination]):
            for band_id in range(self.num_bands):
                slots_state = np.prod(self.channels_map[band_id][path.links_id, :], axis=0).flatten()
                all_channels_capacity = path.channels_modulation_format[band_id] * slots_state * self.basic_channel_capacity  # a
                if np.sum(all_channels_capacity) < self.current_service.bit_rate:
                    continue
                channels, num_required_channels, channels_capacity = self.get_first_available_channels(all_channels_capacity)
                assigned_channels_capacity = np.sum(channels_capacity)
                total_channels_capacity = np.sum(all_channels_capacity)
                m_align = self.cal_spatial_misalignment(path, band_id, channels)
                average_channels_index = np.mean(channels)

                action_id = idp * self.num_bands + band_id
                valid_actions[action_id] = 1
                actions_information[action_id] = [path, band_id, channels]

                spectrum_obs[action_id, :] = np.array([(num_required_channels - 5.5) / 4.5,
                                                       (m_align - 65) / 65,
                                                       (average_channels_index - 53.5) / 53.5,
                                                       (assigned_channels_capacity - 700) / 600,
                                                       (total_channels_capacity - 21600) / 21500
                                                       ])
        if np.all(valid_actions[:-1] == 0):
            valid_actions[-1] = 1

        return valid_actions, actions_information, spectrum_obs

    def cal_spatial_misalignment(self, path, band_id, channels):
        return np.sum(self.channels_map[band_id][path.adjacent_link, :][:, channels])

    def action_masks(self):
        return self.current_service.valid_actions



