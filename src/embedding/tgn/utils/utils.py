import itertools
from collections import deque
from numba import jit

import numpy as np
import torch


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class NeighborFinder:
    def __init__(self, neighbor_buffer_duration_hours, n_edge_features, uniform=False):
        self.n_edge_features = n_edge_features
        self.uniform = uniform
        self.neighbor_buff_duration_ms = neighbor_buffer_duration_hours * 60 * 60 * 1000
        self.adj_list = {}
        self.adj_list_snapshot = None
        self.latest_timestamp = 0

    @jit
    def add_interactions(self, upstreams, downstreams, timestamps, edge_idxs, edge_features):
        unique_node_list = list(set(upstreams) | set(downstreams))
        for node in unique_node_list:
            if node not in self.adj_list:
                self.adj_list[node] = deque()

        batch_nodes = {}

        if len(timestamps) > 0:
            self.latest_timestamp = timestamps[-1]

        for i in range(len(upstreams)):
            us_node = (upstreams[i], downstreams[i], timestamps[i], edge_idxs[i], edge_features[i])
            ds_node = (downstreams[i], upstreams[i], timestamps[i], edge_idxs[i], edge_features[i])

            if upstreams[i] not in batch_nodes:
                batch_nodes[upstreams[i]] = []
            if downstreams[i] not in batch_nodes:
                batch_nodes[downstreams[i]] = []

            batch_nodes[upstreams[i]].append(us_node)
            batch_nodes[downstreams[i]].append(ds_node)

        for node_name, node_obj_list in batch_nodes.items():
            self.adj_list[node_name].extend(node_obj_list)
            while len(self.adj_list[node_name]) > 0 and \
                    self.latest_timestamp - self.adj_list[node_name][0].timestamp > self.neighbor_buff_duration_ms:
                self.adj_list[node_name].popleft()

    @jit
    def get_temporal_neighbor(self, source_nodes, n_neighbors=20):
        all_neighbors = np.zeros((len(source_nodes), n_neighbors))
        all_edge_indices = np.zeros((len(source_nodes), n_neighbors))
        all_timestamps = np.zeros((len(source_nodes), n_neighbors))
        all_edge_features = np.zeros((len(source_nodes), n_neighbors, self.n_edge_features))

        for i, source_node in enumerate(source_nodes):
            source_adj = self.adj_list.get(source_node, deque())

            derived_n_neighbors = min(n_neighbors, len(source_adj))
            if self.uniform and len(source_adj) > 0:
                indices = np.random.choice(len(source_adj), n_neighbors)
                entries = [source_adj[idx] for idx in indices]
            else:
                entries = list(itertools.islice(source_adj, len(source_adj) - derived_n_neighbors, len(source_adj)))

            if entries:
                neighbors = np.array([entry[1] for entry in entries])
                edge_indices = np.array([entry[3] for entry in entries])
                timestamps = np.array([entry[2] for entry in entries])
                edge_features = np.array([entry[4] for entry in entries])

                len_sampled_data = len(neighbors)

                all_neighbors[i, -len_sampled_data:] = neighbors
                all_edge_indices[i, -len_sampled_data:] = edge_indices
                all_timestamps[i, -len_sampled_data:] = timestamps
                all_edge_features[i, -len_sampled_data:, :] = edge_features

        return (
            all_neighbors,
            all_edge_indices,
            all_timestamps,
            all_edge_features.astype(np.float32)
        )

    def reset(self):
        self.adj_list.clear()

    def snapshot(self):
        self.adj_list_snapshot = self.adj_list.copy()

    def restore(self):
        if self.adj_list_snapshot is not None:
            self.adj_list = self.adj_list_snapshot


def get_node_features(source_nodes, n_node_features):
    return np.zeros((len(source_nodes), n_node_features))
