import os
import pickle

import numpy as np
from torch.utils.data import Dataset


class WorkloadPredictionDataset(Dataset):

    def __init__(
            self,
            n_nodes,
            d_embed,
            is_train,
            train_start_minute,
            train_end_minute,
            valid_start_minute,
            valid_end_minute,
            sizes,
            node_id=None
    ):
        self.n_nodes = n_nodes
        self.node_id = node_id

        self.seq_len = sizes[0]
        self.label_len = sizes[1]
        self.pred_len = sizes[2]

        if node_id is None:
            self.n_features = n_nodes + n_nodes * d_embed
            self.n_labels = n_nodes
        else:
            self.n_features = d_embed + 1
            self.n_labels = 1

        self.all_data = np.empty((0, self.n_features), dtype=np.float32)
        self.all_labels = np.empty((0, self.n_labels), dtype=np.float32)

        self._load_data()

        if is_train:
            start_minute = train_start_minute
            end_minute = train_end_minute
        else:
            start_minute = valid_start_minute
            end_minute = valid_end_minute

        self.data = self.all_data[start_minute:end_minute, :]
        self.labels = self.all_labels[start_minute:end_minute, :]

        self._current_idx = 0

    def _load_data(self):
        embedding_dir = os.getenv('EMBEDDING_DIR')

        with open(os.path.join(embedding_dir, 'embeddings_over_time.pickle'), 'rb') as f:
            embeddings = pickle.load(f)

        with open(os.path.join(embedding_dir, 'workloads_over_time.pickle'), 'rb') as f:
            workloads = pickle.load(f)

        node_ids = [self.node_id] if self.node_id is not None else list(range(self.n_nodes))

        for timestep in range(len(embeddings)):
            features_in_current_timestep = np.array([])
            labels_in_current_timestep = np.array([])

            for node_id in node_ids:
                workload = np.array(workloads[timestep][node_id])
                embedding = embeddings[timestep][node_id, :]

                features_in_current_timestep = np.concatenate((features_in_current_timestep, workload, embedding))
                labels_in_current_timestep = np.concatenate((labels_in_current_timestep, workload))

            self.all_data = np.vstack([self.all_data, features_in_current_timestep])
            self.all_labels = np.vstack([self.all_labels, labels_in_current_timestep])

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.labels[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
