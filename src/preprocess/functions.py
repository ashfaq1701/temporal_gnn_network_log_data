import os
import re
import pickle
import random

import pandas as pd


def get_downstream_counts_object():
    filepath = os.path.join(os.getenv('AGGREGATED_STATS_DIR'), "downstream_counts.pickle")
    with open(filepath, 'rb') as f:
        downstream_counts = pickle.load(f)
    return downstream_counts


def get_upstream_counts_object():
    filepath = os.path.join(os.getenv('AGGREGATED_STATS_DIR'), "upstream_counts.pickle")
    with open(filepath, 'rb') as f:
        upstream_counts = pickle.load(f)
    return upstream_counts


def get_rpctype_counts_object():
    filepath = os.path.join(os.getenv('AGGREGATED_STATS_DIR'), "rpctype_counts.pickle")
    with open(filepath, 'rb') as f:
        rpctype_counts = pickle.load(f)
    return rpctype_counts


def get_service_counts_object():
    filepath = os.path.join(os.getenv('AGGREGATED_STATS_DIR'), "service_counts.pickle")
    with open(filepath, 'rb') as f:
        service_counts = pickle.load(f)
    return service_counts


def get_lengths():
    filepath = os.path.join(os.getenv('AGGREGATED_STATS_DIR'), "lengths.pickle")
    with open(filepath, 'rb') as f:
        lengths = pickle.load(f)
    return lengths


def get_all_call_counts(call_counts, from_idx=None, to_idx=None):
    all_counts = {}

    sliced_call_counts = call_counts
    if from_idx is not None and to_idx is not None:
        sliced_call_counts = call_counts[from_idx:to_idx]
    elif from_idx is not None:
        sliced_call_counts = call_counts[from_idx:]
    elif to_idx is not None:
        sliced_call_counts = all_counts[:to_idx]

    for current_counts in sliced_call_counts:
        for node, count in current_counts.items():
            all_counts[node] = all_counts.get(node, 0) + count

    return all_counts


def get_encoded_nodes(upstream_counts, downstream_counts, node_label_encoder, from_idx=None, to_idx=None):
    all_upstream_counts = get_all_call_counts(upstream_counts, from_idx, to_idx)
    all_downstream_counts = get_all_call_counts(downstream_counts, from_idx, to_idx)

    encoded_upstream_nodes = node_label_encoder.transform(list(all_upstream_counts.keys()))
    encoded_downstream_nodes = node_label_encoder.transform(list(all_downstream_counts.keys()))

    return encoded_upstream_nodes, encoded_downstream_nodes


def get_downstream_probabilities(all_downstream_counts):
    total_counts = sum(all_downstream_counts.values())
    downstream_probs = {microservice: count / total_counts for microservice, count in all_downstream_counts.items()}
    return downstream_probs


def get_downstream_microservices(downstream_counts):
    return downstream_counts.keys()


def sample_downstream_microservices(downstream_probs, n=1):
    nodes = list(downstream_probs.keys())
    probabilities = list(downstream_probs.values())

    return random.choices(nodes, probabilities, k=n)


def get_microservice_workload(downstream_counts, microservice):
    workloads = []

    for downstream_count in downstream_counts:
        workloads.append(downstream_count.get(microservice, 0))

    return workloads


def get_all_microservices(downstream_counts, upstream_counts):
    all_downstream_nodes = set()
    all_upstream_nodes = set()

    for downstream_count in downstream_counts:
        all_downstream_nodes = all_downstream_nodes | set(downstream_count.keys())

    for upstream_count in upstream_counts:
        all_upstream_nodes = all_upstream_nodes | set(upstream_count.keys())

    return list(all_upstream_nodes | all_downstream_nodes)


def get_all_rpc_types(rpctype_counts):
    all_rpctype_nodes = set()

    for rpctype_count in rpctype_counts:
        all_rpctype_nodes = all_rpctype_nodes | set(rpctype_count.keys())

    pattern = r"^\d+.*\.*$"
    filtered_rpc_types = [s for s in all_rpctype_nodes if not re.match(pattern, s) and s != 'None']

    return list(filtered_rpc_types)


def get_all_service_counts(service_counts):
    all_service_counts = {}

    for current_service_counts in service_counts:
        for service, count in current_service_counts.items():
            all_service_counts[service] = all_service_counts.get(service, 0) + count

    return all_service_counts


def get_lengths_prefix_sum(lengths):
    prefix_sums = [0] * (len(lengths) + 1)
    for idx, length in enumerate(lengths):
        prefix_sums[idx + 1] = prefix_sums[idx] + length
    return prefix_sums


def get_edge_feature_count():
    final_data_dir = os.getenv('FINAL_DATA_DIR')
    df = pd.read_parquet(os.path.join(final_data_dir, 'data_0.parquet'))
    return len(df.columns) - 4


def get_node_label_encoder():
    filepath = os.path.join(os.getenv('METADATA_DIR'), "node_label_encoder.pickle")
    with open(filepath, 'rb') as f:
        node_label_encoder = pickle.load(f)
    return node_label_encoder
