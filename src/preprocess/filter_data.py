import os

import pandas as pd


def produce_filtered_data(idx, filtered_nodes):
    input_path = os.path.join(os.getenv('FINAL_DATA_DIR'), f'data_{idx}.parquet')
    output_path = os.path.join(os.getenv('FILTERED_DATA_DIR'), f'data_{idx}.parquet')

    df = pd.read_parquet(input_path)
    filtered_df = df[df['u'].isin(filtered_nodes) & df['i'].isin(filtered_nodes)]

    filtered_df.to_parquet(output_path)
    print(f'Finished processing {output_path}')
