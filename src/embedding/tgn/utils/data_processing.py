import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CombinedPandasDatasetFromDirectory(Dataset):
    def __init__(self, file_dir, start_file_idx, end_file_idx, batch_size=2000, neighbor_finder=None, logger=None):
        self.logger = logger

        self.file_dir = file_dir
        self.file_paths = [
            os.path.join(file_dir, f'data_{idx}.parquet')
            for idx in range(start_file_idx, end_file_idx)
        ]
        self.batch_size = batch_size
        self.neighbor_finder = neighbor_finder

        self.current_file_index = 0
        self.num_batches_in_file = 0
        self.last_file_end_batch_idx = -1
        self.current_df = pd.DataFrame()

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.next_file_future: Future = None

        self._load_initial_file()

    def _load_initial_file(self):
        self.current_df = self._load_file(0)
        self.num_batches_in_file = math.ceil(len(self.current_df) / self.batch_size)
        if len(self.file_paths) > 1:
            self.next_file_future = self.executor.submit(self._load_file, 1)

    def _load_file(self, file_index):
        if self.logger is not None:
            self.logger.info(f'Loading file {self.file_paths[file_index]} in dataset')
        df = pd.read_parquet(self.file_paths[file_index])
        df['rt'] = df['rt'].fillna(df['rt'].median())
        return df

    def _load_next_file_in_background(self):
        if self.current_file_index < len(self.file_paths) - 1:
            self.current_file_index += 1
            self.current_df = self.next_file_future.result()
            self.num_batches_in_file = math.ceil(len(self.current_df) / self.batch_size)
            if self.current_file_index < len(self.file_paths) - 1:
                self.next_file_future = self.executor.submit(self._load_file, self.current_file_index + 1)
        else:
            raise IndexError("No more files to load")

    def __len__(self):
        total_rows = sum(pd.read_parquet(file_path).shape[0] for file_path in self.file_paths)
        return math.ceil(total_rows / self.batch_size)

    def __getitem__(self, idx):
        if idx - self.last_file_end_batch_idx > self.num_batches_in_file:
            self._load_next_file_in_background()
            self.last_file_end_batch_idx = idx - 1

        file_batch_index = idx - self.last_file_end_batch_idx - 1
        start_row = file_batch_index * self.batch_size
        end_row = min(start_row + self.batch_size, len(self.current_df))
        batch_data = self.current_df.iloc[start_row:end_row]

        upstreams = batch_data[['u']].values.flatten().astype(np.int32)
        downstreams = batch_data[['i']].values.flatten().astype(np.int32)
        timestamps = batch_data[['ts']].values.flatten().astype(np.int64)
        edge_indices = batch_data[['idx']].values.flatten().astype(np.int64)
        edge_features = batch_data.iloc[:, 4:].values.astype(np.int32)

        return upstreams, downstreams, timestamps, edge_indices, edge_features

    def add_batch_to_neighbor_finder(self, batch):
        upstreams, downstreams, timestamps, edge_indices, edge_features = batch
        if self.neighbor_finder is not None:
            self.neighbor_finder.add_interactions(upstreams, downstreams, timestamps, edge_indices, edge_features)

    def reset(self):
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_file_index = 0
        self.last_file_end_batch_idx = -1
        self.current_df = self._load_file(0)
        self.num_batches_in_file = math.ceil(len(self.current_df) / self.batch_size)
        if len(self.file_paths) > 1:
            self.next_file_future = self.executor.submit(self._load_file, 1)


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
