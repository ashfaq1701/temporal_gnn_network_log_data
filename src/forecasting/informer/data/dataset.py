import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from src.forecasting.informer.data.custom_std_scaler import CustomStandardScaler
from src.forecasting.informer.utils.timefeatures import time_encode


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
            seq_len,
            label_len,
            pred_len,
            use_temporal_embedding=True,
            node_id=None
    ):
        self.n_nodes = n_nodes
        self.node_id = node_id

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.workload_scaler = CustomStandardScaler()

        self.use_temporal_embedding = use_temporal_embedding

        if use_temporal_embedding:
            self.embedding_width = d_embed
        else:
            self.embedding_width = 0

        if node_id is None:
            self.n_features = n_nodes + n_nodes * self.embedding_width
            self.n_labels = n_nodes
        else:
            self.n_features = self.embedding_width + 1
            self.n_labels = 1

        self.all_data = np.empty((0, self.n_features), dtype=np.float32)
        self.all_labels = np.empty((0, self.n_labels), dtype=np.float32)

        self._load_data()

        all_timesteps = np.arange(1, self.all_data.shape[0] + 1, dtype=np.float32)
        self.all_timesteps = time_encode(all_timesteps)

        if is_train:
            start_minute = train_start_minute
            end_minute = train_end_minute
        else:
            start_minute = valid_start_minute
            end_minute = valid_end_minute

        self.data = self.all_data[start_minute:end_minute, :]
        self.labels = self.all_labels[start_minute:end_minute, :]
        self.timesteps = self.all_timesteps[start_minute:end_minute, :]

        self._current_idx = 0

    def _load_data(self):
        embedding_dir = os.getenv('EMBEDDING_DIR')

        with open(os.path.join(embedding_dir, 'embeddings_over_time.pickle'), 'rb') as f:
            embeddings = pickle.load(f)

        with open(os.path.join(embedding_dir, 'workloads_over_time.pickle'), 'rb') as f:
            workloads = pickle.load(f)
            workloads = np.array(workloads)

        scaled_workloads = self.workload_scaler.fit_transform(workloads)

        node_ids = [self.node_id] if self.node_id is not None else list(range(self.n_nodes))

        self.all_data = np.zeros((len(scaled_workloads), self.n_features), dtype=np.float32)
        self.all_labels = np.zeros((len(scaled_workloads), self.n_labels), dtype=np.float32)

        for timestep in range(len(embeddings)):
            col_idx_data = 0
            col_idx_labels = 0

            for node_id in node_ids:
                workload = scaled_workloads[timestep][node_id]
                embedding = embeddings[timestep][node_id, :]

                self.all_data[timestep, col_idx_data] = workload
                col_idx_data += 1

                if self.use_temporal_embedding:
                    self.all_data[timestep, col_idx_data:col_idx_data + self.embedding_width] = embedding
                    col_idx_data += self.embedding_width

                self.all_labels[timestep, col_idx_labels] = workload
                col_idx_labels += 1

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError('No more data')

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.labels[r_begin:r_end]

        seq_x_mark = self.timesteps[s_begin:s_end]
        seq_y_mark = self.timesteps[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.workload_scaler.inverse_transform(data)

    def get_feature_and_label_count(self):
        return self.n_features, self.n_labels
