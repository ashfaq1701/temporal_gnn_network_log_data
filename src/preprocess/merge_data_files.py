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

    merged_df = None
    dataframe_batch = []

    for idx in range(20160):
        filepath = os.path.join(input_dir, f'data_{idx}.parquet')
        df = pd.read_parquet(filepath)
        dataframe_batch.append(df)

        if len(dataframe_batch) == BATCH_SIZE:
            if merged_df is None:
                merged_df = pd.concat(dataframe_batch, ignore_index=True)
            else:
                merged_df = pd.concat([merged_df] + dataframe_batch, ignore_index=True)

            dataframe_batch = []

        print(f'Read file {filepath}')

    features_df = merged_df[['u', 'i', 'ts', 'label', 'idx']]
    edges_df = merged_df.iloc[:, 4:]
    nodes_df = np.zeros((num_nodes, num_node_features), dtype=np.float32)

    output_dir = os.getenv('MERGED_DATA_DIR')

    features_df.to_csv(os.path.join(output_dir, 'ml_alibaba.csv'))
    np.save(os.path.join(output_dir, 'ml_alibaba.npy'), edges_df)
    np.save(os.path.join(output_dir, 'ml_alibaba_node.npy'), nodes_df)
