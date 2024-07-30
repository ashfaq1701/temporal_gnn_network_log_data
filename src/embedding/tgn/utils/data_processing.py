import math
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ParquetDataset(Dataset):
    def __init__(self, file_dir, start_file_idx, end_file_idx, batch_size=2000):
        self.file_dir = file_dir
        self.file_paths = [
            os.path.join(file_dir, f'data_{idx}.parquet')
            for idx in range(start_file_idx, end_file_idx)
        ]
        self.batch_size = batch_size
        self.current_file_index = 0
        self.current_df = self._load_next_file()
        self.num_batches_in_file = math.ceil(len(self.current_df) / self.batch_size)

    def _load_next_file(self):
        if self.current_file_index >= len(self.file_paths):
            raise IndexError("No more files to load")
        file_path = self.file_paths[self.current_file_index]
        self.current_file_index += 1
        df = pd.read_parquet(file_path)
        df['rt'] = df['rt'].fillna(df['rt'].median())
        return df

    def __len__(self):
        total_rows = sum(pd.read_parquet(file_path).shape[0] for file_path in self.file_paths)
        return math.ceil(total_rows / self.batch_size)

    def __getitem__(self, idx):
        file_batch_index = idx % self.num_batches_in_file
        if idx // self.num_batches_in_file > self.current_file_index:
            self.current_df = self._load_next_file()
            self.num_batches_in_file = math.ceil(len(self.current_df) / self.batch_size)

        start_row = file_batch_index * self.batch_size
        end_row = min(start_row + self.batch_size, len(self.current_df))
        batch_data = self.current_df.iloc[start_row:end_row]
        return (
            batch_data[['u']].values,
            batch_data[['i']].values,
            batch_data[['ts']].values,
            batch_data[['idx']].values,
            batch_data.iloc[:, 4:].values
        )


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
