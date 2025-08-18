from typing import List
import gym
import numpy as np

from .DRL_RMBSA import DRL_RMBSA


class DRL_RMBSA_PCA(DRL_RMBSA):
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
            mean_service_holding_time=mean_service_holding_time,
            mean_service_inter_arrival_time=mean_service_inter_arrival_time,
            bands=bands,
            use_mask=use_mask,
            node_request_probabilities=node_request_probabilities,
            ksp_node_lists=ksp_node_lists,
            modulation_table=modulation_table,
            GSNR_table=GSNR_table,
            GSNR_record_episode=GSNR_record_episode,
            seed=seed,
        )

        shape = self.max_hops * self.k_paths + self.k_paths * self.num_bands * 5 + self.k_paths

        self.observation_space = gym.spaces.Box(
            low=-100, high=100, dtype=np.float64, shape=(shape,)
        )
        self.observation_space.seed(self.rand_seed)

    def step(self, action: int):
        self.current_service.best_action = True if action in self.current_service.lowest_cost_action else 0
        return super().step(action)

    def reward(self):
        if self.current_service.accepted:
            extra_reward = 1 if self.current_service.best_action else 0.9
            return extra_reward
        else:
            return -1

    def preprocessing(self):
        valid_actions = np.zeros([self.k_paths * self.num_bands + self.use_mask], dtype=bool)
        spectrum_obs = np.full((self.k_paths * self.num_bands + 1, 5), fill_value=-1.0)
        actions_information = [None for i in range(self.action_space.n)]
        cost_map = np.full((self.k_paths * self.num_bands + self.use_mask), dtype=np.float64, fill_value=500)
        for idp, path in enumerate(
                self.k_shortest_paths[self.current_service.source, self.current_service.destination]):

            total_channels_capacity_path = np.sum(
                np.concatenate([np.prod(self.channels_map[band_id][path.links_id, :], axis=0).flatten() *
                                path.channels_modulation_format[band_id] for band_id in range(self.num_bands)]))
            spectrum_obs[-1,idp] = (total_channels_capacity_path - 700) / 600
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

                cost = 1 / (total_channels_capacity_path)
                cost_map[action_id] = cost

                spectrum_obs[action_id, :] = np.array([(num_required_channels - 5.5) / 4.5,
                                                       (m_align - 65) / 65,
                                                       (average_channels_index - 53.5) / 53.5,
                                                       (assigned_channels_capacity - 700) / 600,
                                                       (total_channels_capacity - 21600) / 21500,
                                                       ])

        self.current_service.lowest_cost_action = np.where(cost_map == np.min(cost_map))[0]
        if self.use_mask and np.all(valid_actions[:-1] == 0):
            valid_actions[-1] = 1

        return valid_actions, actions_information, spectrum_obs


