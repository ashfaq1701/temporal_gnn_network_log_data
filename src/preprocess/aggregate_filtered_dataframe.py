import os

import pandas as pd


def aggregate_filtered_dataframe(idx):
    input_dir = os.getenv('FILTERED_DATA_DIR')
    filepath = os.path.join(input_dir, f'data_{idx}.parquet')
    df = pd.read_parquet(filepath)
    u_counts = df['u'].value_counts().to_dict()
    i_counts = df['i'].value_counts().to_dict()
    print(f'Computed stats for {filepath}')
    return idx, u_counts, i_counts, len(df)
