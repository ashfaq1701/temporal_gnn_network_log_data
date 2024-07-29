import os

import pandas as pd

from src.embedding.tgn.utils.data_processing import compute_time_statistics


def compute_time_statistics_for_file(i):
    input_dir = os.getenv('FINAL_DATA_DIR')

    input_filepath = os.path.join(input_dir, f'data_{i}.parquet')
    df = pd.read_parquet(input_filepath)
    sources = df['u'].to_numpy()
    destinations = df['i'].to_numpy()
    ts = df['ts'].to_numpy()

    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(
        sources,
        destinations,
        ts
    )

    print(f'Computed statistics for {input_filepath}')
    return i, mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
