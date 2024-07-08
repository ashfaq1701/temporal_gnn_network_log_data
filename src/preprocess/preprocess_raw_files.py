import os

import pandas as pd

from src.utils import download_file, if_file_exists, extract_tar_gz

_base_url = 'https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces/CallGraph'


def download_and_process_callgraph(file_id):
    filename = f'CallGraph_{file_id}.tar.gz'
    file_url = f'{_base_url}/{filename}'
    download_dir = os.getenv('RAW_FILE_DIR')
    tmp_dir = os.getenv('EXTRACTED_FILE_DIR')

    if not if_file_exists(download_dir, filename):
        download_file(file_url, download_dir)

    extracted_filename = f'CallGraph_{file_id}.csv'
    extracted_filepath = os.path.join(tmp_dir, extracted_filename)
    extract_tar_gz(
        os.path.join(download_dir, filename),
        tmp_dir
    )

    try:
        read_and_process_file(extracted_filepath, file_id)
    except Exception as e:
        print(f"Exception processing the callgraph file: {extracted_filepath}, error: {e}")

    os.remove(extracted_filepath)


def read_and_process_file(extracted_filepath, file_id):
    print(f'Processing file {extracted_filepath}')
    raw_df = pd.read_csv(extracted_filepath, on_bad_lines='skip')
    df_with_unknowns_removed = raw_df.replace('UNKNOWN', None)

    first_timestamp = str(df_with_unknowns_removed.iloc[0]['timestamp'])
    if not first_timestamp.isdigit() and first_timestamp.startswith('T_'):
        df_with_unknowns_removed = df_with_unknowns_removed.shift(periods=1, axis=1)
        df_with_unknowns_removed['timestamp'] = df_with_unknowns_removed.index

    mandatory_cols = ['timestamp', 'traceid', 'service', 'um', 'dm']
    cleaned_df = df_with_unknowns_removed.dropna(subset=mandatory_cols)

    selected_df = cleaned_df[
        ['timestamp', 'traceid', 'service', 'um', 'dm', 'rt']
    ]

    selected_df['timestamp'] = selected_df['timestamp'].astype(int)
    selected_df['rt'] = pd.to_numeric(selected_df['rt'], errors='coerce')

    sorted_df = selected_df.sort_values(by='timestamp')

    output_dir = os.getenv('OUTPUT_DIR')
    output_filepath = os.path.join(output_dir, f'CallGraph_{file_id}.parquet')

    sorted_df.to_parquet(output_filepath)
