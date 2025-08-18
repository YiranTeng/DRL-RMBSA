import copy
import functools
import bisect
import heapq
import logging
import math
import functools
import itertools
import time
import pickle
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple, List, Union

import gym
import networkx as nx
import numpy as np

from utils import Path, Service
from .optical_network_env import OpticalNetworkEnv


class MBEON(OpticalNetworkEnv):
    metadata = {
        "metrics": [
            "service_blocking_rate",
            "episode_service_blocking_rate",
            "bit_rate_blocking_rate",
            "episode_bit_rate_blocking_rate",
        ]
    }

    def __init__(
        self,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        load: float = 10,
        mean_service_holding_time: float = 10800.0,
        num_spectrum_resources: int = 100,
        node_request_probabilities = None,
        ksp_node_lists=None,
        modulation_table=None,
        GSNR_table=None,
        GSNR_record_episode=None,
        bit_rates = [100, 200, 400, 1000],
        bands: tuple = ('L', 'C', 'S'),
        seed = None,
        reset = True,
        channel_width: float = 12.5
    ):

        super().__init__(
            topology,
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            channel_width=channel_width,
        )

        # specific attributes for elastic optical networks
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0

        self.bit_rates = np.array(bit_rates)
        self.bit_rate_probabilities = [
            1.0 / len(bit_rates) for x in range(len(bit_rates))
        ]
        self.bit_rate_function = functools.partial(
            self.rng.choices, self.bit_rates, self.bit_rate_probabilities, k=1
        )

        self.basic_channel_capacity = 100
        self.num_links = self.topology.number_of_edges()
        self.bands = bands
        self.bands_slots = dict(L=80, C=80, S=108, E=160)
        self.bands_channels_range = list(itertools.accumulate([0] + [self.bands_slots[band] for band in self.bands]))
        self.generate_modulation_and_GSNR_table(ksp_node_lists, modulation_table, GSNR_table)
        self.GSNR_record_episode = GSNR_record_episode

        # defining the observation and action spaces
        self.logger = logging.getLogger("mbeonenv")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                "Logging is enabled for DEBUG which generates a large number of \
                messages. \
                Set it to INFO if DEBUG is not necessary."
            )
        self._new_service = False
        if reset:
            self.reset(only_episode_counters=False)

    def step(self, action: Union[None, List]):
        if action is not None:
            selected_path, band_id, channels = action

            assert self.is_action_feasible(
                    selected_path,
                    band_id,
                    channels)

            self._provision_path(
                selected_path,
                band_id,
                channels
            )
            self.current_service.accepted = True
            self._add_release(self.current_service)

        else:
            self.current_service.accepted = False

        # More statistics
        self.services_processed += 1
        self.episode_services_processed += 1
        self.bit_rate_requested += self.current_service.bit_rate
        self.episode_bit_rate_requested += self.current_service.bit_rate

        # Get the value of the action
        reward = self.reward()

        if self.episode_services_processed == self.episode_length:
            info = {
                "service_blocking_rate": (self.services_processed - self.services_accepted)
                / self.services_processed,
                "episode_service_blocking_rate": (
                    self.episode_services_processed - self.episode_services_accepted
                )
                / self.episode_services_processed,
                "bit_rate_blocking_rate": (
                    self.bit_rate_requested - self.bit_rate_provisioned
                )
                / self.bit_rate_requested,
                "episode_bit_rate_blocking_rate": (
                    self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
                )
                / self.episode_bit_rate_requested
        }

        else:
            info = {}

        self._new_service = False
        self._next_service()

        return (
            self.observation(),
            reward,
            self.episode_services_processed == self.episode_length,
            info,
        )
    
    def reset(self, only_episode_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        if self.GSNR_record_episode is not None and self.services_processed >= self.GSNR_record_episode * self.episode_length:
            self.GSNR_record = True

        if only_episode_counters:
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.channels_map = [np.ones((self.num_links, self.bands_slots[band])) for band in self.bands]
        self.channels_state = []
        self.lightpath_GSNR = []
        self.modulation_usage = np.zeros((6,), dtype=np.int)
        self.band_usage = np.zeros((len(self.bands),), dtype=np.int)
        self.path_usage = np.zeros((self.k_paths,), dtype=np.int)
        self.GSNR_record = False
        self.heat_map = [np.zeros((self.num_links, self.bands_slots[band])) for band in self.bands]


        self._new_service = False
        self._next_service()

        return self.observation()

    def generate_modulation_and_GSNR_table(self, ksp_node_lists, modulation_table, GSNR_table):
        sd_pairs, node_lists = ksp_node_lists
        for idx, (node_pair, path_list) in enumerate(self.k_shortest_paths.items()):
            if idx % 2 == 1:
                self.k_shortest_paths[node_pair] = self.k_shortest_paths[(node_pair[1], node_pair[0])]
                continue
            node_pair_array = np.array([int(x) for x in node_pair])
            matches = np.all(sd_pairs == node_pair_array, axis=1)
            matching_rows = np.where(matches)[0]
            assert matching_rows.size == 1
            for idp, path in enumerate(path_list):
                if not np.all(np.array([int(x) for x in path.node_list]) == node_lists[matching_rows[0]][idp]):
                    path.node_list = [str(element) for element in node_lists[matching_rows[0]][idp]]
                channels_modulation_format = []
                channels_GSNR = []
                for idb in range(len(self.bands)):
                    channels_modulation_format.append(modulation_table[matching_rows[0], idp, self.bands_channels_range[idb] : self.bands_channels_range[idb + 1]].astype(np.int32))
                    channels_GSNR.append(GSNR_table[matching_rows[0], idp, self.bands_channels_range[idb] : self.bands_channels_range[idb + 1]].astype(np.int32))
                path.channels_modulation_format = channels_modulation_format
                path.channels_GSNR = channels_GSNR

    def is_action_feasible(
        self, path: Path, band_id: int, channels: np.ndarray
    ) -> bool:
        if np.max(channels) >= self.bands_slots[self.bands[band_id]]:
            return False
        if np.any(self.channels_map[band_id][path.links_id, :][:, channels] == 0):
            return False
        if np.sum(path.channels_modulation_format[band_id][channels] * self.basic_channel_capacity) < self.current_service.bit_rate:
            return False
        return True

    def _provision_path(
        self, path: Path, band_id: int, channels: np.ndarray,
    ):
        assert np.all(self.channels_map[band_id][path.links_id, :][:, channels] == 1)

        self.channels_map[band_id][np.ix_(path.links_id, channels)] = 0

        self.current_service.path = path
        self.current_service.band_id = band_id
        self.current_service.channels = channels

        if self.GSNR_record:
            self.lightpath_GSNR.append(path.channels_GSNR[band_id][channels])
            self.band_usage[band_id] += 1
            self.path_usage[path.path_id % self.k_paths] += 1
            mf_channels_1 = path.channels_modulation_format[band_id][channels[:-1]]
            self.modulation_usage[mf_channels_1 - 1] += 1
            last_channels = path.channels_modulation_format[band_id][channels[-1]]
            mf_channels_2 = path.channels_modulation_format[band_id][last_channels] - (
                np.sum(path.channels_modulation_format[band_id][channels]) - int(
                self.current_service.bit_rate / self.basic_channel_capacity))
            self.modulation_usage[mf_channels_2 - 1] += 1

            self.heat_map[band_id][np.ix_(path.links_id, channels)] += self.current_service.holding_time

        self.services_accepted += 1
        self.episode_services_accepted += 1
        self.bit_rate_provisioned += self.current_service.bit_rate
        self.episode_bit_rate_provisioned += self.current_service.bit_rate

    def _release_path(self, service: Service):
        assert np.all(
            self.channels_map[service.band_id][service.path.links_id, :][:, service.channels] == 0)
        self.channels_map[service.band_id][np.ix_(service.path.links_id, service.channels)] = 1

    def _next_service(self):
        if self._new_service:
            return

        interval = self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        at = self.current_time + interval
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        # generate the bit rate according to the selection adopted
        bit_rate = self.bit_rate_function()[0]

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop

        self.current_service = Service(
            self.episode_services_processed,
            src,
            src_id,
            destination=dst,
            destination_id=dst_id,
            arrival_time=at,
            holding_time=ht,
            bit_rate=bit_rate,
        )
        self._new_service = True

    def get_channels_state(self):
        return self.channels_state

    def get_lightpath_mean_GSNR(self):
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            assert time >= self.current_time
            compensate_time = time - self.current_time
            self.heat_map[service_to_release.band_id][np.ix_(service_to_release.path.links_id, service_to_release.channels)] -= compensate_time
            if len(self._events) == 0:
                break
        heat_map = np.concatenate([self.heat_map[i] for i in range(len(self.bands))], axis=1)
        heatmap = heat_map / self.current_time
        with open('heatmap_KSP_MinMaxF.pkl', 'wb') as file:  # 'wb' 表示以二进制写模式打开文件
            pickle.dump(heatmap, file)
        return np.mean(np.concatenate(self.lightpath_GSNR)), self.band_usage / np.sum(self.band_usage),\
        self.modulation_usage / np.sum(self.modulation_usage), self.path_usage / np.sum(self.path_usage)