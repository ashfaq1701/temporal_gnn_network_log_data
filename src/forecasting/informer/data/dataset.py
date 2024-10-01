import numpy as np
from torch.utils.data import Dataset

from src.forecasting.informer.utils.timefeatures import time_encode


class WorkloadPredictionDataset(Dataset):

    def __init__(
            self,
            workloads,
            embeddings,
            start_minute,
            end_minute,
            seq_len,
            label_len,
            pred_len,
            workload_scaler
    ):
        self.n_nodes = workloads.shape[-1]

        self.workloads = workloads
        self.embeddings = embeddings

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.workload_scaler = workload_scaler

        if embeddings is not None:
            self.embedding_width = embeddings.shape[-1]
        else:
            self.embedding_width = 0

        self.n_features = self.n_nodes + self.n_nodes * self.embedding_width
        self.n_labels = self.n_nodes

        self.data = np.empty((workloads.shape[0], self.n_features), dtype=np.float32)
        self.labels = np.empty((workloads.shape[0], self.n_labels), dtype=np.float32)
        timesteps = np.arange(start_minute, end_minute + 1, dtype=np.float32)
        self.timesteps = time_encode(timesteps)

        self._load_data()

        self._current_idx = 0

    def _load_data(self):
        self.all_data = np.zeros((len(self.workloads), self.n_features), dtype=np.float32)
        self.all_labels = np.zeros((len(self.workloads), self.n_labels), dtype=np.float32)

        for timestep in range(self.workloads.shape[0]):
            col_idx_data = 0
            col_idx_labels = 0

            for i in range(self.n_nodes):
                workload = self.workloads[timestep][i]

                self.data[timestep, col_idx_data] = workload
                col_idx_data += 1

                if self.embeddings is not None:
                    embedding = self.embeddings[timestep, i, :]
                    self.data[timestep, col_idx_data:col_idx_data + self.embedding_width] = embedding
                    col_idx_data += self.embedding_width

                self.labels[timestep, col_idx_labels] = workload
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
