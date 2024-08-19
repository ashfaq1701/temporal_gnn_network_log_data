import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.embedding.tgn.model.tgn import TGN
from src.embedding.tgn.utils.data_processing import CombinedPandasDatasetFromDirectory
from src.embedding.tgn.utils.utils import get_unique_latest_nodes_with_indices, get_future_workloads, \
    get_past_workloads
from src.preprocess.compute_time_statistics import compute_time_shifts_for_n_days
from src.preprocess.functions import get_edge_feature_count, get_filtered_workload_counts, get_filtered_node_counts

sys.path.append(os.path.abspath('./build'))
import neighbor_finder


def precompute_temporal_embedding(args):
    link_prediction_batch_size = args.link_prediction_bs
    num_neighbors = args.n_degree
    num_heads = args.n_head
    drop_out = args.drop_out
    gpu = args.gpu
    num_layer = args.n_layer
    use_memory = args.use_memory
    message_dim = args.message_dim
    memory_dim = args.memory_dim
    n_past = args.n_past
    n_future = args.n_future

    results_dir = os.getenv('RESULTS_DIR')

    Path(os.path.join(results_dir, "saved_models")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "saved_checkpoints")).mkdir(parents=True, exist_ok=True)

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(results_dir, 'log/{}.log').format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    data_directory = os.getenv('FILTERED_DATA_DIR')

    neighbor_buffer_duration_hours = int(os.getenv('NEIGHBOR_BUFFER_DURATION_HOURS'))
    n_edge_features = get_edge_feature_count()
    ngh_finder = neighbor_finder.NeighborFinder(neighbor_buffer_duration_hours, n_edge_features, args.uniform)

    full_dataset = CombinedPandasDatasetFromDirectory(
        data_directory,
        0,
        20_160,
        link_prediction_batch_size,
        ngh_finder,
        logger
    )

    n_nodes = get_filtered_node_counts()

    workloads = get_filtered_workload_counts()

    # Set device
    if torch.cuda.is_available():
        device_string = 'cuda:{}'.format(gpu)
    elif torch.backends.mps.is_available():
        device_string = 'mps'
    else:
        device_string = 'cpu'

        num_cores = os.cpu_count()
        print(f'Num CPU cores: {num_cores}')
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(num_cores)

    device = torch.device(device_string)
    logger.info('Device: {}'.format(device))

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_shifts_for_n_days(14)

    embedding_buffer = np.zeros((n_nodes, memory_dim), dtype=np.float32)
    embeddings_over_time = []
    past_workloads = []
    future_workloads = []

    tgn = TGN(neighbor_finder=ngh_finder, n_node_features=memory_dim, n_nodes=n_nodes,
              n_edge_features=n_edge_features, device=device, n_layers=num_layer, n_heads=num_heads,
              dropout=drop_out, use_memory=use_memory, message_dimension=message_dim, memory_dimension=memory_dim,
              memory_update_at_start=not args.memory_update_at_end, embedding_module_type=args.embedding_module,
              message_function=args.message_function, aggregator_type=args.aggregator, n_neighbors=num_neighbors,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message)
    tgn = tgn.to(device)

    current_minute = 0

    for batch in full_dataset:
        sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, \
            edge_features_batch, current_file_end = batch

        with torch.no_grad():
            _, destination_embeddings, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                           destinations_batch,
                                                                           destinations_batch,
                                                                           timestamps_batch,
                                                                           edge_idxs_batch,
                                                                           num_neighbors)
            destination_embeddings_np = destination_embeddings.detach().cpu().numpy()

        full_dataset.add_batch_to_neighbor_finder(batch[:-1])

        nodes_with_latest_indices = get_unique_latest_nodes_with_indices(destinations_batch, timestamps_batch)
        latest_destination_embeddings = destination_embeddings_np[nodes_with_latest_indices[:, 1]]
        nodes = nodes_with_latest_indices[:, 0]
        for idx in range(len(nodes)):
            embedding_buffer[nodes[idx], :] = latest_destination_embeddings[idx, :]

        if current_file_end:
            past_workloads_current_minute = get_past_workloads(
                np.arange(0, n_nodes),
                current_minute,
                workloads,
                n_past
            )
            future_workloads_current_minute = get_future_workloads(
                np.arange(0, n_nodes),
                current_minute,
                workloads,
                n_future
            )

            embeddings_over_time.append(embedding_buffer.copy())
            past_workloads.append(past_workloads_current_minute)
            future_workloads.append(future_workloads_current_minute)

            current_minute += 1

    embedding_dir = os.getenv('EMBEDDING_DIR')
    pickle.dump(embeddings_over_time, open(os.path.join(embedding_dir, 'embeddings_over_time.pickle'), "wb"))
    pickle.dump(past_workloads, open(os.path.join(embedding_dir, 'past_workloads_over_time.pickle'), "wb"))
    pickle.dump(future_workloads, open(os.path.join(embedding_dir, 'future_workloads_over_time.pickle'), "wb"))
