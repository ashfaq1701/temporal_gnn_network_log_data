import os
import pandas as pd
import pickle

from src.utils import count_trimmed_values


def aggregate_dataframe(file_idx):
    input_dir = os.getenv('PER_MINUTE_OUTPUT_DIR')
    output_dir = os.getenv('STATS_DIR')

    input_filepath = os.path.join(input_dir, f'CallGraph_{file_idx}.parquet')
    df = pd.read_parquet(input_filepath)

    upstream_counts = count_trimmed_values(df, 'um')
    downstram_counts = count_trimmed_values(df, 'dm')
    df_len = len(df)

    stats = {
        'upstream_counts': upstream_counts,
        'downstream_counts': downstram_counts,
        'length': df_len
    }

    output_filepath = os.path.join(output_dir, f'CallGraph_stats_{file_idx}.pickle')
    with open(output_filepath, 'wb') as f:
        pickle.dump(stats, f)

    print(f'Saved stats to {output_filepath}')



