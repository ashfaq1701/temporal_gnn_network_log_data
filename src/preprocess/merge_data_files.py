import os

import numpy as np
import pandas as pd

from src.preprocess.functions import get_filtered_node_label_encoder

BATCH_SIZE = 1000


def merge_data_files():
    filtered_node_features_encoder = get_filtered_node_label_encoder()
    num_nodes = len(filtered_node_features_encoder.classes_)
    num_node_features = int(os.getenv('N_NODE_FEATURES'))
    input_dir = os.getenv('FILTERED_DATA_DIR')
    dataframes = []

    for idx in range(20160):
        filepath = os.path.join(input_dir, f'data_{idx}.parquet')
        dataframes.append(pd.read_parquet(filepath))

    merged_df = combine_dataframes_in_batches(dataframes, batch_size=1000)

    features_df = merged_df[['u', 'i', 'ts', 'label', 'idx']]
    edges_df = merged_df.iloc[:, 4:]
    nodes_df = np.zeros((num_nodes, num_node_features), dtype=np.float32)

    output_dir = os.getenv('MERGED_DATA_DIR')

    features_df.to_csv(os.path.join(output_dir, 'ml_alibaba.csv'))
    np.save(os.path.join(output_dir, 'ml_alibaba.npy'), edges_df)
    np.save(os.path.join(output_dir, 'ml_alibaba_node.npy'), nodes_df)


def combine_dataframes_in_batches(dfs, batch_size=100):
    combined_df = pd.DataFrame()
    for i in range(0, len(dfs), batch_size):
        batch_dfs = dfs[i:i+batch_size]
        combined_df = pd.concat([combined_df] + batch_dfs, ignore_index=True)
    return combined_df
