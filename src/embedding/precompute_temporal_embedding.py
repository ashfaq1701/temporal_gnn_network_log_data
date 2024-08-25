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
from src.embedding.tgn.utils.utils import get_unique_latest_nodes_with_indices, get_workloads_at_time
from src.preprocess.compute_time_statistics import compute_time_shifts_for_n_days
from src.preprocess.functions import get_edge_feature_count, get_filtered_workload_counts, get_filtered_nodes_count

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

    n_nodes = get_filtered_nodes_count()

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
    workloads_by_minutes = []

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

    model_path = os.path.join(results_dir, f'saved_models/{args.prefix}-alibaba.pth')
    tgn.load_state_dict(torch.load(model_path))
    tgn.eval()
    logger.info('TGN models loaded and eval mode started.')

    current_minute = 0

    for batch in full_dataset:
        sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, \
            edge_features_batch, current_file_end = batch

        with torch.no_grad():
            source_embeddings, destination_embeddings, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                           destinations_batch,
                                                                                           destinations_batch,
                                                                                           timestamps_batch,
                                                                                           edge_idxs_batch,
                                                                                           num_neighbors)
            source_embeddings_np = source_embeddings.detach().cpu().numpy()
            destination_embeddings_np = destination_embeddings.detach().cpu().numpy()
            all_embeddings = np.concatenate((destination_embeddings_np, source_embeddings_np))

        full_dataset.add_batch_to_neighbor_finder(batch[:-1])

        nodes_with_latest_indices = get_unique_latest_nodes_with_indices(
            np.concatenate((destinations_batch, sources_batch)),
            np.concatenate((timestamps_batch, timestamps_batch))
        )
        latest_node_embeddings = all_embeddings[nodes_with_latest_indices[:, 1]]
        nodes = nodes_with_latest_indices[:, 0]
        for idx in range(len(nodes)):
            embedding_buffer[nodes[idx], :] = latest_node_embeddings[idx, :]

        if current_file_end:
            workloads_current_minute = get_workloads_at_time(
                np.arange(0, n_nodes),
                current_minute,
                workloads
            )

            embeddings_over_time.append(embedding_buffer.copy())
            workloads_by_minutes.append(workloads_current_minute)

            current_minute += 1

    embedding_dir = os.getenv('EMBEDDING_DIR')
    pickle.dump(embeddings_over_time, open(os.path.join(embedding_dir, 'embeddings_over_time.pickle'), "wb"))
    pickle.dump(workloads_by_minutes, open(os.path.join(embedding_dir, 'workloads_over_time.pickle'), "wb"))
