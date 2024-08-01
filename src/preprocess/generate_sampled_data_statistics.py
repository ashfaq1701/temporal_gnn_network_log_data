import os
import pandas as pd


def generate_statistics_for_sampled_file(idx):
    input_dir = os.getenv('SAMPLED_DATA_DIR')
    filepath = os.path.join(input_dir, f'data_{idx}.parquet')
    df = pd.read_parquet(filepath)
    u_counts = df['u'].value_counts().to_dict()
    i_counts = df['i'].value_counts().to_dict()
    len_df = len(df)
    print(f'Computed statistics for {filepath}')
    return idx, u_counts, i_counts, len_df
