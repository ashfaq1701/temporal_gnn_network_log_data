import os
import re
import pickle
import random


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


def get_all_downstream_call_counts(downstream_counts):
    all_downstream_counts = {}

    for current_downstream_counts in downstream_counts:
        for downstream, count in current_downstream_counts.items():
            all_downstream_counts[downstream] = all_downstream_counts.get(downstream, 0) + count

    return all_downstream_counts


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
