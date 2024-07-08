import argparse
import concurrent.futures
import os
import pickle
from dotenv import load_dotenv

from src.preprocess.aggregate_dataframe import aggregate_dataframe
from src.preprocess.get_per_minute_dataframes import break_file_into_per_minute_dataframes
from src.preprocess.preprocess_raw_files import download_and_process_callgraph


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
    all_stats = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(aggregate_dataframe, i) for i in range(start_idx, end_idx)]

        for future in concurrent.futures.as_completed(futures):
            try:
                file_idx, stats = future.result()
                all_stats[file_idx] = stats
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    ordered_stats = [stats for _, stats in sorted(all_stats.items())]
    output_dir = os.getenv('STATS_DIR')
    output_filepath = os.path.join(output_dir, f'all_stats.pickle')
    with open(output_filepath, 'wb') as f:
        pickle.dump(ordered_stats, f)


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
            produce_per_minute_files(
                args.start_index,
                args.end_index
            )
        case 'aggregate':
            aggregate_dataframes(args.start_index, args.end_index)
        case _:
            raise ValueError(f'Invalid task: {args.task}')
