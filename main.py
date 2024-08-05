import argparse
import concurrent.futures
import os
import pickle

from dotenv import load_dotenv
from src.embedding.train_self_supervised import train_link_prediction_model
from src.preprocess.aggregate_dataframe import aggregate_dataframe, get_stats
from src.preprocess.aggregate_filtered_dataframe import aggregate_filtered_dataframe
from src.preprocess.compute_graph import compute_downstream_graph_for_file
from src.preprocess.compute_seasonality import compute_seasonality_of_microservices
from src.preprocess.compute_time_statistics import compute_time_statistics_for_file
from src.preprocess.filter_data import produce_filtered_data
from src.preprocess.filter_nodes import filter_nodes_k_neighbors, save_filtered_label_encoder
from src.preprocess.functions import get_lengths, get_lengths_prefix_sum, get_downstream_counts_object, \
    get_upstream_counts_object, get_rpctype_counts_object, get_all_microservices, get_all_rpc_types, \
    get_node_label_encoder, get_filtered_nodes, get_filtered_node_label_encoder
from src.preprocess.get_per_minute_dataframes import break_file_into_per_minute_dataframes
from src.preprocess.preprocess_raw_files import download_and_process_callgraph
from src.preprocess.produce_final_format_data import get_label_encoder, get_one_hot_encoder, store_encoders, \
    produce_final_format_for_file
from src.utils import get_files_in_directory_with_ext


def preprocess(start_day, start_hour, end_day, end_hour):
    start_minute = start_day * 24 * 60 + start_hour * 60
    end_minute = end_day * 24 * 60 + end_hour * 60

    start_file_idx = int(start_minute / 3)
    end_file_idx = int(end_minute / 3) - 1

    n_workers = int(os.getenv('N_WORKERS_PREPROCESSING'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(download_and_process_callgraph, i) for i in range(start_file_idx, end_file_idx + 1)]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")


def produce_per_minute_files(start_idx, end_idx):
    n_workers = int(os.getenv('N_WORKERS_PREPROCESSING'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(break_file_into_per_minute_dataframes, i) for i in range(start_idx, end_idx)]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")


def aggregate_dataframes(start_idx, end_idx):
    n_workers = int(os.getenv('N_WORKERS_PREPROCESSING'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(aggregate_dataframe, i) for i in range(start_idx, end_idx)]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")


def merge_stats():
    input_dir = os.getenv('STATS_DIR')
    file_names = get_files_in_directory_with_ext(input_dir, '.pickle')
    print(f'Total files: {len(file_names)}')
    n_workers = int(os.getenv('N_WORKERS_PREPROCESSING'))

    all_upstream_counts = {}
    all_downstream_counts = {}
    all_rpctype_counts = {}
    all_service_counts = {}
    all_lengths = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(get_stats, filename) for filename in file_names]

        for future in concurrent.futures.as_completed(futures):
            try:
                file_idx, current_stats = future.result()

                all_upstream_counts[file_idx] = current_stats['upstream_counts']
                all_downstream_counts[file_idx] = current_stats['downstream_counts']
                all_rpctype_counts[file_idx] = current_stats['rpctype_counts']
                all_service_counts[file_idx] = current_stats['service_counts']
                all_lengths[file_idx] = current_stats['length']
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    ordered_upstream_counts = [
        upstream_counts
        for _, upstream_counts in sorted(all_upstream_counts.items())
    ]
    ordered_downstream_counts = [
        downstream_counts
        for _, downstream_counts in sorted(all_downstream_counts.items())
    ]
    ordered_rpctype_counts = [
        rpctype_counts
        for _, rpctype_counts in sorted(all_rpctype_counts.items())
    ]
    ordered_service_counts = [
        service_counts
        for _, service_counts in sorted(all_service_counts.items())
    ]
    ordered_lengths = [
        length
        for _, length in sorted(all_lengths.items())
    ]

    output_dir = os.getenv('AGGREGATED_STATS_DIR')

    with open(os.path.join(output_dir, 'upstream_counts.pickle'), 'wb') as f:
        pickle.dump(ordered_upstream_counts, f)

    with open(os.path.join(output_dir, 'downstream_counts.pickle'), 'wb') as f:
        pickle.dump(ordered_downstream_counts, f)

    with open(os.path.join(output_dir, 'rpctype_counts.pickle'), 'wb') as f:
        pickle.dump(ordered_rpctype_counts, f)

    with open(os.path.join(output_dir, 'service_counts.pickle'), 'wb') as f:
        pickle.dump(ordered_service_counts, f)

    with open(os.path.join(output_dir, 'lengths.pickle'), 'wb') as f:
        pickle.dump(ordered_lengths, f)

    print('Stored all aggregated data files.')


def produce_final_format_data(start_idx, end_idx):
    lengths = get_lengths()
    length_prefix_sums = get_lengths_prefix_sum(lengths)

    downstream_counts = get_downstream_counts_object()
    upstream_counts = get_upstream_counts_object()
    rpctype_counts = get_rpctype_counts_object()

    all_nodes = get_all_microservices(downstream_counts, upstream_counts)
    node_label_encoder = get_label_encoder(all_nodes)

    all_rpc_types = get_all_rpc_types(rpctype_counts)
    rpc_type_one_hot_encoder = get_one_hot_encoder(all_rpc_types)

    n_workers = int(os.getenv('N_WORKERS_PREPROCESSING'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                produce_final_format_for_file,
                i,
                length_prefix_sums[i],
                node_label_encoder,
                rpc_type_one_hot_encoder
            )
            for i in range(start_idx, end_idx)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    store_encoders(node_label_encoder, rpc_type_one_hot_encoder)


def compute_graphs():
    def merge_graphs(graph, combined_graph):
        for outer, inner_graph in graph.items():
            combined_inner_graph = combined_graph.get(outer, {})

            for inner, count in inner_graph.items():
                combined_inner_graph[inner] = combined_inner_graph.get(inner, 0) + count

            combined_graph[outer] = combined_inner_graph

    combined_i_graph = {}
    combined_u_graph = {}

    n_workers = int(os.getenv('N_WORKERS_PREPROCESSING'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(compute_downstream_graph_for_file, idx) for idx in range(20160)]

        for future in concurrent.futures.as_completed(futures):
            _, i_graph, u_graph = future.result()

            merge_graphs(i_graph, combined_i_graph)
            merge_graphs(u_graph, combined_u_graph)

    output_dir = os.getenv('AGGREGATED_STATS_DIR')
    with open(os.path.join(output_dir, 'downstream_graph.pickle'), 'wb') as f:
        pickle.dump(combined_i_graph, f)

    with open(os.path.join(output_dir, 'upstream_graph.pickle'), 'wb') as f:
        pickle.dump(combined_u_graph, f)


def compute_seasonality():
    all_seasonality = compute_seasonality_of_microservices()
    output_dir = os.getenv('AGGREGATED_STATS_DIR')
    with open(os.path.join(output_dir, 'all_seasonality.pickle'), 'wb') as f:
        pickle.dump(all_seasonality, f)


def filter_nodes():
    nodes = os.getenv('MICROSERVICE_LIST').split(',')
    k = int(os.getenv('K_NEIGHBOR_FILTER'))
    filtered_node_list = filter_nodes_k_neighbors(nodes, k)
    save_filtered_label_encoder(filtered_node_list)
    with open(os.path.join(os.getenv('AGGREGATED_STATS_DIR'), 'filtered_nodes.pickle'), 'wb') as f:
        pickle.dump(filtered_node_list, f)


def filter_data_files():
    label_encoder = get_node_label_encoder()
    filtered_label_encoder = get_filtered_node_label_encoder()
    filtered_nodes = get_filtered_nodes()
    encoded_filtered_nodes = label_encoder.transform(filtered_nodes)
    filtered_node_set = set(encoded_filtered_nodes)

    n_workers = int(os.getenv('N_WORKERS_PREPROCESSING'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(produce_filtered_data, idx, filtered_node_set, label_encoder, filtered_label_encoder)
            for idx in range(20160)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")


def compute_all_time_statistics_for_files():
    all_mean_time_shift_src = {}
    all_std_time_shift_src = {}
    all_mean_time_shift_dst = {}
    all_std_time_shift_dst = {}
    all_lengths = {}

    n_workers = int(os.getenv('N_WORKERS_PREPROCESSING'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(compute_time_statistics_for_file, i) for i in range(20160)]

        for future in concurrent.futures.as_completed(futures):
            try:
                idx, mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst, df_len = \
                    future.result()
                all_mean_time_shift_src[idx] = mean_time_shift_src
                all_std_time_shift_src[idx] = std_time_shift_src
                all_mean_time_shift_dst[idx] = mean_time_shift_dst
                all_std_time_shift_dst[idx] = std_time_shift_dst
                all_lengths[idx] = df_len
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    ordered_mean_time_shift_src = [
        mean_time_shift_src
        for _, mean_time_shift_src in sorted(all_mean_time_shift_src.items())
    ]
    ordered_std_time_shift_src = [
        std_time_shift_src
        for _, std_time_shift_src in sorted(all_std_time_shift_src.items())
    ]
    ordered_mean_time_shift_dst = [
        mean_time_shift_dst
        for _, mean_time_shift_dst in sorted(all_mean_time_shift_dst.items())
    ]
    ordered_std_time_shift_dst = [
        std_time_shift_dst
        for _, std_time_shift_dst in sorted(all_std_time_shift_dst.items())
    ]
    ordered_lengths = [df_len for _, df_len in sorted(all_lengths.items())]

    stats_for_all_files = {
        'all_mean_time_shift_src': ordered_mean_time_shift_src,
        'all_std_time_shift_src': ordered_std_time_shift_src,
        'all_mean_time_shift_dst': ordered_mean_time_shift_dst,
        'all_std_time_shift_dst': ordered_std_time_shift_dst,
        'all_lengths': ordered_lengths
    }

    metadata_dir = os.getenv('METADATA_DIR')
    with open(os.path.join(metadata_dir, 'all_time_statistics.pickle'), 'wb') as f:
        pickle.dump(stats_for_all_files, f)


def compute_filtered_stats():
    n_workers = int(os.getenv('N_WORKERS_PREPROCESSING'))

    all_u_counts = {}
    all_i_counts = {}
    all_lens = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(aggregate_filtered_dataframe, i) for i in range(20160)]
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, u_counts, i_counts, len_df = future.result()
                all_u_counts[idx] = u_counts
                all_i_counts[idx] = i_counts
                all_lens[idx] = len_df
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    ordered_u_counts = [u_counts for _, u_counts in sorted(all_u_counts.items())]
    ordered_i_counts = [i_counts for _, i_counts in sorted(all_i_counts.items())]
    ordered_lens = [df_len for _, df_len in sorted(all_lens.items())]

    stats_obj = {
        'upstream_counts': ordered_u_counts,
        'downstream_counts': ordered_i_counts,
        'lens': ordered_lens
    }

    stats_dir = os.getenv('AGGREGATED_STATS_DIR')
    with open(os.path.join(stats_dir, 'filtered_counts.pickle'), 'wb') as f:
        pickle.dump(stats_obj, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alibaba Graph Training.")

    # Add arguments to the parser
    parser.add_argument('--task', type=str, required=True, help='The task to execute.')
    parser.add_argument('--start_day', type=int, help='The start day.')
    parser.add_argument('--start_hour', type=int, help='The start hour.')
    parser.add_argument('--end_day', type=int, help='The end day.')
    parser.add_argument('--end_hour', type=int, help='The end hour.')
    parser.add_argument('--start_index', type=int, help='Index of starting file.')
    parser.add_argument('--end_index', type=int, help='Index of ending file.')
    parser.add_argument('--checkpoints', type=int, nargs='+', help='Checkpoints to store time statistics.')

    # TGN Arguments
    parser.add_argument('--bs', type=int, default=2000, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')
    parser.add_argument('--use_memory', action='store_true',
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                       'aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                    'each user')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true',
                        help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--dyrep', action='store_true',
                        help='Whether to run the dyrep model')

    # Parse the command-line arguments
    args = parser.parse_args()

    load_dotenv()

    match args.task:
        case 'preprocess_raw_files':
            preprocess(
                args.start_day,
                args.start_hour,
                args.end_day,
                args.end_hour
            )
        case 'produce_per_minute_files':
            produce_per_minute_files(args.start_index, args.end_index)
        case 'aggregate':
            aggregate_dataframes(args.start_index, args.end_index)
        case 'merge_stats':
            merge_stats()
        case 'produce_final_format_data':
            produce_final_format_data(args.start_index, args.end_index)
        case 'compute_graphs':
            compute_graphs()
        case 'compute_seasonality':
            compute_seasonality()
        case 'filter_nodes':
            filter_nodes()
        case 'produce_filtered_data':
            filter_data_files()
        case 'produce_filtered_stats':
            compute_filtered_stats()
        case 'compute_time_statistics':
            compute_all_time_statistics_for_files()
        case 'train_link_prediction':
            train_link_prediction_model(args)
        case _:
            raise ValueError(f'Invalid task: {args.task}')
