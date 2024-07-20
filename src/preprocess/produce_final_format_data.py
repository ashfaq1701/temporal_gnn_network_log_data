import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

_encoded_rpc_types = {}
_encoded_services = {}


def produce_final_format_for_file(file_idx, edge_start_idx, node_label_encoder, rpc_type_one_hot_encoder):

    input_dir = os.getenv('PER_MINUTE_OUTPUT_DIR')
    output_dir = os.getenv('FINAL_DATA_DIR')

    filepath = os.path.join(input_dir, f'CallGraph_{file_idx}.parquet')
    df = pd.read_parquet(filepath)

    transformed_df = pd.DataFrame({
        'u': node_label_encoder.transform(df['um']),
        'i': node_label_encoder.transform(df['dm']),
        'ts': df['timestamp'],
        'label': 0,
        'idx': range(edge_start_idx + 1, edge_start_idx + len(df) + 1)
    })

    new_index = pd.RangeIndex(start=edge_start_idx, stop=edge_start_idx + len(df))
    transformed_df.index = new_index

    output_filepath = os.path.join(output_dir, f'data_{file_idx}.parquet')
    transformed_df.to_parquet(output_filepath)
    print(f'Stored file {output_filepath}')

    get_and_save_edge_features(df, file_idx, rpc_type_one_hot_encoder, edge_start_idx)


def store_encoders(node_label_encoder, rpc_type_one_hot_encoder):
    metadata_path = os.getenv('METADATA_DIR')
    node_label_encoder_file_path = os.path.join(metadata_path, 'node_label_encoder.pickle')
    with open(node_label_encoder_file_path, 'wb') as f:
        pickle.dump(node_label_encoder, f)

    node_one_hot_encoder = get_nodes_one_hot_encoder(node_label_encoder)
    node_one_hot_encoder_file_path = os.path.join(metadata_path, 'node_one_hot_encoder.pickle')
    with open(node_one_hot_encoder_file_path, 'wb') as f:
        pickle.dump(node_one_hot_encoder, f)

    rpc_type_one_hot_encoder_file_path = os.path.join(metadata_path, 'rpc_type_one_hot_encoder.pickle')
    with open(rpc_type_one_hot_encoder_file_path, 'wb') as f:
        pickle.dump(rpc_type_one_hot_encoder, f)

    print(f'Stored all encoders')


def get_encoded_rpc_type(rpc_type, rpc_type_encoder):
    if rpc_type in _encoded_rpc_types:
        return _encoded_rpc_types[rpc_type]

    encoded_type = rpc_type_encoder.transform([[rpc_type]])[0]
    _encoded_rpc_types[rpc_type] = encoded_type
    return encoded_type


def get_encoded_service(service, service_encoder):
    if service in _encoded_services:
        return _encoded_services[service]

    encoded_service = service_encoder.transform([[service]])[0]
    _encoded_services[service] = encoded_service
    return encoded_service


def get_and_save_edge_features(df, idx, rpc_type_encoder, edge_start_idx):
    rt_col = df[['rt']]
    one_hot_encoded_rpc_type = rpc_type_encoder.transform(df[['rpctype']])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_rpc_type)
    new_df = pd.concat([
        rt_col.reset_index(drop=True),
        one_hot_encoded_df.reset_index(drop=True)
    ], axis=1)

    new_index = pd.RangeIndex(start=edge_start_idx, stop=edge_start_idx + len(df))
    new_df.index = new_index

    edge_attrs_dir = os.getenv('EDGE_ATTRS_DIR')
    edge_attrs_file_path = os.path.join(edge_attrs_dir, f'edge_attrs_{idx}.parquet')
    new_df.to_parquet(edge_attrs_file_path)


def get_label_encoder(nodes):
    label_encoder = LabelEncoder()
    label_encoder.fit(nodes)
    return label_encoder


def get_nodes_one_hot_encoder(label_encoder):
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit(label_encoder.classes_.reshape(-1, 1))
    return one_hot_encoder


def get_one_hot_encoder(items):
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit([[item] for item in items])
    return one_hot_encoder
