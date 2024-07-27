import argparse
import concurrent.futures
import os
import pickle

from dotenv import load_dotenv

from src.preprocess.aggregate_dataframe import aggregate_dataframe, get_stats
from src.preprocess.compute_time_statistics import compute_all_time_statistics
from src.preprocess.functions import get_lengths, get_lengths_prefix_sum, get_downstream_counts_object, \
    get_upstream_counts_object, get_rpctype_counts_object, get_all_microservices, get_all_rpc_types
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
        case 'compute_time_statistics':
            compute_all_time_statistics(args.checkpoints)
        case _:
            raise ValueError(f'Invalid task: {args.task}')
