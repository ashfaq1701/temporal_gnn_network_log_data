import os
import pickle
import pandas as pd
import numpy as np

from src.utils import combine_means_and_stds


def compute_all_time_statistics(checkpoints):
    input_dir = os.getenv('FINAL_DATA_DIR')

    mean_src = []
    std_src = []
    mean_dst = []
    std_dst = []
    counts = []

    for i in range(20160):
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

        mean_src.append(mean_time_shift_src)
        std_src.append(std_time_shift_src)
        mean_dst.append(mean_time_shift_dst)
        std_dst.append(std_time_shift_dst)
        counts.append(len(df))

        print(f'Computed statistics for {input_filepath}')

    output_dir = os.getenv('METADATA_DIR')

    for day in checkpoints:
        checkpoint_minute = day * 24 * 60
        combined_mean_src, combined_std_src = combine_means_and_stds(
            mean_src[:checkpoint_minute],
            std_src[:checkpoint_minute],
            counts[:checkpoint_minute]
        )
        combined_mean_dst, combined_std_dst = combine_means_and_stds(
            mean_dst[:checkpoint_minute],
            std_dst[:checkpoint_minute],
            counts[:checkpoint_minute]
        )

        time_stats = {
            'mean_src': combined_mean_src,
            'std_src': combined_std_src,
            'mean_dst': combined_mean_dst,
            'std_dst': combined_std_dst
        }

        output_filepath = os.path.join(output_dir, f'time_statistics_{day}')

        with open(output_filepath, 'wb') as f:
            pickle.dump(time_stats, f)


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
