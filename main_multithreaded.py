import argparse
import concurrent.futures
import os

from dotenv import load_dotenv

from src.preprocess_raw_files import download_and_process_callgraph


def preprocess(start_day, start_hour, end_day, end_hour):
    load_dotenv()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alibaba Graph Training.")

    # Add arguments to the parser
    parser.add_argument('--task', type=str, required=True, help='The task to execute.')
    parser.add_argument('--start_day', type=int, required=True, help='The start day.')
    parser.add_argument('--start_hour', type=int, required=True, help='The start hour.')
    parser.add_argument('--end_day', type=int, required=True, help='The end day.')
    parser.add_argument('--end_hour', type=int, required=True, help='The end hour.')

    # Parse the command-line arguments
    args = parser.parse_args()

    match args.task:
        case 'preprocess_raw_files':
            preprocess(
                args.start_day,
                args.start_hour,
                args.end_day,
                args.end_hour
            )
        case _:
            raise ValueError(f'Invalid task: {args.task}')
