import os
import pickle

import pandas as pd

from src.embedding.tgn.utils.data_processing import compute_time_statistics
from src.utils import combine_means_and_stds


def compute_time_statistics_for_file(i):
    input_dir = os.getenv('SAMPLED_DATA_DIR')

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
    return i, mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst, len(df)


def compute_time_shifts_for_n_days(days):
    metadata_dir = os.getenv('METADATA_DIR')

    with open(os.path.join(metadata_dir, 'all_time_statistics.pickle'), 'rb') as f:
        time_stats = pickle.load(f)

    all_mean_time_shift_src = time_stats['all_mean_time_shift_src']
    all_std_time_shift_src = time_stats['all_std_time_shift_src']
    all_mean_time_shift_dst = time_stats['all_mean_time_shift_dst']
    all_std_time_shift_dst = time_stats['all_std_time_shift_dst']
    all_lengths = time_stats['all_lengths']

    total_minutes = days * 24 * 60

    combined_mean_src, combined_std_src = combine_means_and_stds(
        all_mean_time_shift_src[:total_minutes],
        all_std_time_shift_src[:total_minutes],
        all_lengths[:total_minutes]
    )
    combined_mean_dst, combined_std_dst = combine_means_and_stds(
        all_mean_time_shift_dst[:total_minutes],
        all_std_time_shift_dst[:total_minutes],
        all_lengths[:total_minutes]
    )

    return combined_mean_src, combined_std_src, combined_mean_dst, combined_std_dst
