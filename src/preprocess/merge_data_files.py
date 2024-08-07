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
        df = pd.read_parquet(filepath)
        dataframes.append(df)
        print(f'Read {filepath}')

    merged_df = pd.concat(dataframes, ignore_index=True)

    features_df = merged_df[['u', 'i', 'ts', 'label', 'idx']]
    edges_df = merged_df.iloc[:, 4:]
    nodes_df = np.zeros((num_nodes, num_node_features), dtype=np.float32)

    output_dir = os.getenv('MERGED_DATA_DIR')

    features_df.to_csv(os.path.join(output_dir, 'ml_alibaba.csv'))
    np.save(os.path.join(output_dir, 'ml_alibaba.npy'), edges_df)
    np.save(os.path.join(output_dir, 'ml_alibaba_node.npy'), nodes_df)
