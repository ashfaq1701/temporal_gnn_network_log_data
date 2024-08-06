import os

import numpy as np
import random
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(train_days, valid_days):
    ### Load data and train val test split
    input_dir = os.getenv('MERGED_DATA_DIR')
    graph_df = pd.read_csv(os.path.join(input_dir, 'ml_alibaba.csv'))
    edge_features = np.load(os.path.join(input_dir, 'ml_alibaba.npy'))
    node_features = np.load(os.path.join(input_dir, 'ml_alibaba_node.npy'))

    val_start_time = train_days * 24 * 60 * 60 * 1000
    val_end_time = (train_days + valid_days) * 24 * 60 * 60 * 1000

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    random.seed(2020)

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = timestamps <= val_start_time

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    val_mask = np.logical_and(timestamps > val_start_time, timestamps <= val_end_time)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    return full_data, node_features, edge_features, train_data, val_data


def get_data(train_days, valid_days):
    ### Load data and train val test split
    input_dir = os.getenv('MERGED_DATA_DIR')
    graph_df = pd.read_csv(os.path.join(input_dir, 'ml_alibaba.csv'))
    edge_features = np.load(os.path.join(input_dir, 'ml_alibaba.npy'))
    node_features = np.load(os.path.join(input_dir, 'ml_{}_node.npy'))

    val_start_time = train_days * 24 * 60 * 60 * 1000
    val_end_time = (train_days + valid_days) * 24 * 60 * 60 * 1000

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2020)

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = timestamps <= val_start_time

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    val_mask = np.logical_and(timestamps > val_start_time, timestamps <= val_end_time)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))

    return node_features, edge_features, full_data, train_data, val_data


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
