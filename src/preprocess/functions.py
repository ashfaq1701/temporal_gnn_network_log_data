import os
import pickle
import random


def get_stats_object():
    filepath = os.path.join(os.getenv('AGGREGATED_STATS_DIR'), "all_stats.pickle")
    with open(filepath, 'rb') as f:
        stats = pickle.load(f)
    return stats


def get_lengths(stats):
    return [stat['length'] for stat in stats]


def get_all_downstream_call_counts(stats):
    all_downstream_counts = {}

    for current_stat in stats:
        for downstream, count in current_stat['downstream_counts'].items():
            all_downstream_counts[downstream] = all_downstream_counts.get(downstream, 0) + count

    return all_downstream_counts


def get_downstream_probabilities(downstream_counts):
    total_counts = sum(downstream_counts.values())
    downstream_probs = {microservice: count / total_counts for microservice, count in downstream_counts.items()}
    return downstream_probs


def get_downstream_microservices(downstream_counts):
    return downstream_counts.keys()


def sample_downstream_microservices(downstream_probs, n=1):
    nodes = list(downstream_probs.keys())
    probabilities = list(downstream_probs.values())

    return random.choices(nodes, probabilities, k=n)


def get_microservice_workload(stats, microservice):
    workloads = []

    for stat in stats:
        downstream_counts = stat['downstream_counts']
        workloads.append(downstream_counts.get(microservice, 0))

    return workloads


def get_all_microservices(stats):
    all_downstream_nodes = set()
    all_upstream_nodes = set()

    for current_stat in stats:
        all_downstream_nodes = all_downstream_nodes | set(current_stat['downstream_counts'].keys())
        all_upstream_nodes = all_upstream_nodes | set(current_stat['upstream_counts'].keys())

    return list(all_upstream_nodes | all_downstream_nodes)
