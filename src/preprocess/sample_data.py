import os

import pandas as pd


def sample_data(file_idx):
    sample_percentage = float(os.getenv('SAMPLE_PERCENTAGE'))

    def sample_group(group):
        n = max(int(len(group) * sample_percentage), 1)  # Ensure at least one sample per group
        return group.sample(n=n)

    input_dir = os.getenv('FINAL_DATA_DIR')
    output_dir = os.getenv('SAMPLED_DATA_DIR')

    input_df = pd.read_parquet(os.path.join(input_dir, f'data_{file_idx}.parquet'))
    sampled_df = input_df.groupby('i', group_keys=False).apply(sample_group).reset_index(drop=True)
    sampled_df.to_parquet(os.path.join(output_dir, f'data_{file_idx}.parquet'))
