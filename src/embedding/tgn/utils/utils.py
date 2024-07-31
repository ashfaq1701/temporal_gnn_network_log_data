import itertools
from collections import deque
import multiprocessing as mp
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


class NeighborInfo:
    def __init__(self, node, neighbor, timestamp, edge_idx, edge_features):
        self.node = node
        self.neighbor = neighbor
        self.timestamp = timestamp
        self.edge_idx = edge_idx
        self.edge_features = edge_features

    def __lt__(self, other):
        return self.timestamp < other.timestamp


class NeighborFinder:
    def __init__(self, max_time_in_seconds, n_edge_features, uniform=False):
        self.n_edge_features = n_edge_features
        self.uniform = uniform
        self.max_time_in_milliseconds = max_time_in_seconds * 1000
        self.adj_list = {}
        self.adj_list_snapshot = None
        self.latest_timestamp = 0

    def add_interactions(self, upstreams, downstreams, timestamps, edge_idxs, edge_features):
        data = zip(upstreams, downstreams, timestamps, edge_idxs, edge_features)
        for row in data:
            self._process_interaction(row)

    def _process_interaction(self, interaction):
        us, ds, ts, edge_idx, edge_feat = interaction
        self.latest_timestamp = max(self.latest_timestamp, ts)
        self.add_neighbor_to_node(us, ds, ts, edge_idx, edge_feat)
        self.add_neighbor_to_node(ds, us, ts, edge_idx, edge_feat)

    def add_neighbor_to_node(self, node, neighbor, timestamp, edge_idx, edge_features):
        if node not in self.adj_list:
            self.adj_list[node] = deque()

        neighbor_info = NeighborInfo(node, neighbor, timestamp, edge_idx, edge_features)

        self.adj_list[node].append(neighbor_info)

        while len(self.adj_list[node]) > 0 and \
                self.latest_timestamp - self.adj_list[node][0].timestamp > self.max_time_in_milliseconds:
            self.adj_list[node].popleft()

    def get_temporal_neighbor(self, source_nodes, n_neighbors=20):
        all_neighbors = []
        all_edge_indices = []
        all_timestamps = []
        all_edge_features = []

        for source_node in source_nodes:
            source_adj = self.adj_list.get(source_node, deque())

            neighbors = np.zeros(n_neighbors)
            edge_indices = np.zeros(n_neighbors)
            timestamps = np.zeros(n_neighbors)
            edge_features = np.zeros((n_neighbors, self.n_edge_features))

            derived_n_neighbors = n_neighbors
            if len(source_adj) > 0 and n_neighbors > 0:
                if self.uniform:
                    indices = np.random.randint(0, len(source_adj), n_neighbors)
                    entries = [source_adj[idx] for idx in indices]
                else:
                    derived_n_neighbors = min(n_neighbors, len(source_adj))
                    entries = list(itertools.islice(source_adj, len(source_adj) - derived_n_neighbors, len(source_adj)))

                neighbors[-derived_n_neighbors:] = np.array([entry.neighbor for entry in entries])
                edge_indices[-derived_n_neighbors:] = np.array([entry.edge_idx for entry in entries])
                timestamps[-derived_n_neighbors:] = np.array([entry.timestamp for entry in entries])
                edge_features[-derived_n_neighbors:, :] = np.array([entry.edge_features for entry in entries])

            all_neighbors.append(neighbors)
            all_edge_indices.append(edge_indices)
            all_timestamps.append(timestamps)
            all_edge_features.append(edge_features)

        return np.array(all_neighbors), np.array(all_edge_indices), np.array(all_timestamps), np.array(all_edge_features)

    def reset(self):
        self.adj_list.clear()

    def snapshot(self):
        self.adj_list_snapshot = self.adj_list.copy()

    def restore(self):
        if self.adj_list_snapshot is not None:
            self.adj_list = self.adj_list_snapshot


def get_node_features(source_nodes, n_nodes):
    one_hot_array = np.zeros((len(source_nodes), n_nodes))
    for i, node in enumerate(source_nodes):
        one_hot_array[i, node] = 1
    return one_hot_array
