import os

import numpy as np
import pandas as pd
import pickle
from src.preprocess.functions import get_stats_object, get_all_microservices, get_all_rpc_types, get_all_services
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

_encoded_rpc_types = {}
_encoded_services = {}


def produce_final_format_data():
    stats = get_stats_object()
    all_nodes = get_all_microservices(stats)
    node_label_encoder = get_label_encoder(all_nodes)

    all_rpc_types = get_all_rpc_types(stats)
    all_services = get_all_services(stats)
    rpc_type_one_hot_encoder = get_one_hot_encoder(all_rpc_types)
    service_one_hot_encoder = get_one_hot_encoder(all_services)

    input_dir = os.getenv('PER_MINUTE_OUTPUT_DIR')
    output_dir = os.getenv('FINAL_DATA_DIR')

    running_total = 0

    for idx in range(20160):
        filepath = os.path.join(input_dir, f'CallGraph_{idx}.parquet')
        df = pd.read_parquet(filepath)

        transformed_df = pd.DataFrame({
            'u': node_label_encoder.transform(df['um']),
            'i': node_label_encoder.transform(df['dm']),
            'ts': df['timestamp'],
            'label': 0,
            'idx': range(running_total + 1, running_total + len(df) + 1)
        })

        new_index = pd.RangeIndex(start=running_total, stop=running_total + len(df))
        transformed_df.index = new_index

        running_total += len(df)

        output_filepath = os.path.join(output_dir, f'data_{idx}.parquet')
        transformed_df.to_parquet(output_filepath)
        print(f'Stored file {output_filepath}')

        get_and_save_edge_features(df, idx, rpc_type_one_hot_encoder, service_one_hot_encoder)

    metadata_path = os.getenv('METADATA_DIR')
    node_label_encoder_file_path = os.path.join(metadata_path, 'node_label_encoder.pickle')
    with open(node_label_encoder_file_path, 'wb') as f:
        pickle.dump(node_label_encoder, f)

    node_one_hot_encoder = get_nodes_one_hot_encoder(node_label_encoder)
    node_one_hot_encoder_file_path = os.path.join(metadata_path, 'node_one_hot_encoder.pickle')
    with open(node_one_hot_encoder_file_path, 'wb') as f:
        pickle.dump(node_one_hot_encoder, f)

    node_feats = get_node_features(node_label_encoder.classes_, node_one_hot_encoder)
    node_features_file_path = os.path.join(metadata_path, 'node_features.npy')
    np.save(node_features_file_path, node_feats)

    rpc_type_one_hot_encoder_file_path = os.path.join(metadata_path, 'rpc_type_one_hot_encoder.pickle')
    with open(rpc_type_one_hot_encoder_file_path, 'wb') as f:
        pickle.dump(rpc_type_one_hot_encoder, f)

    service_one_hot_encoder_file_path = os.path.join(metadata_path, 'service_one_hot_encoder.pickle')
    with open(service_one_hot_encoder_file_path, 'wb') as f:
        pickle.dump(service_one_hot_encoder, f)

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


def get_and_save_edge_features(df, idx, rpc_type_encoder, service_encoder):
    df['rt'] = df['rt'].replace(0, None)
    mean_rt_value = df['rt'].mean()
    df['rt'].fillna(mean_rt_value, inplace=True)

    formatted_rows = []

    for _, row in df.iterrows():
        rt = row['rt']
        rpc_type = None if row['rpctype'] is None else row['rpctype'].strip()
        service = None if row['service'] is None else row['service'].strip()

        formatted_row = np.array([rt])
        formatted_rpc_type = get_encoded_rpc_type(rpc_type, rpc_type_encoder)
        formatted_service = get_encoded_service(service, service_encoder)

        formatted_row = np.concatenate((formatted_row, formatted_rpc_type, formatted_service))
        formatted_rows.append(formatted_row)

    edge_attrs_dir = os.getenv('EDGE_ATTRS_DIR')
    edge_attrs_file_path = os.path.join(edge_attrs_dir, f'edge_attrs_{idx}.npy')
    np.save(edge_attrs_file_path, np.array(formatted_rows))
    print(f'Stored edge attributes {edge_attrs_file_path}')


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


def get_node_features(nodes, one_hot_encoder):
    return one_hot_encoder.transform([[class_name] for class_name in nodes])
