import os
import pandas as pd

MINUTES_PER_FILE = 3


def break_file_into_per_minute_dataframes(file_idx):
    input_dir = os.getenv('OUTPUT_DIR')
    output_dir = os.getenv('PER_MINUTE_OUTPUT_DIR')

    input_filepath = os.path.join(input_dir, f'CallGraph_{file_idx}.parquet')
    input_df = pd.read_parquet(input_filepath)

    file_start_timestamp = file_idx * MINUTES_PER_FILE * 60 * 1000
    for start_minute in range(0, MINUTES_PER_FILE):
        current_start_timestamp = file_start_timestamp + start_minute * 60 * 1000
        current_end_timestamp = file_start_timestamp + (start_minute + 1) * 60 * 1000
        current_file_idx = file_idx * MINUTES_PER_FILE + start_minute

        minute_level_df = input_df[
            (input_df['timestamp'] >= current_start_timestamp) &
            (input_df['timestamp'] < current_end_timestamp)
        ]

        output_filepath = os.path.join(output_dir, f'CallGraph_{current_file_idx}.parquet')
        minute_level_df.to_parquet(output_filepath)
        print(f'Stored file {output_filepath}')

