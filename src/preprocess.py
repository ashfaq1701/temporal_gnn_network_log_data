import os

import pandas as pd

from src.utils import download_file, if_file_exists, get_filename_from_url, extract_tar_gz

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
    read_and_process_file(extracted_filepath, file_id)
    os.remove(extracted_filepath)


def read_and_process_file(extracted_filepath, file_id):
    print(f'Processing file {extracted_filepath}')
    raw_df = pd.read_csv(extracted_filepath, on_bad_lines='skip')
    df_with_unknowns_removed = raw_df.replace('UNKNOWN', None)
    mandatory_cols = ['timestamp', 'traceid', 'service', 'um', 'dm']
    cleaned_df = df_with_unknowns_removed.dropna(subset=mandatory_cols)

    cleaned_df['timestamp'] = cleaned_df['timestamp'].astype(int)
    cleaned_df['rt'] = pd.to_numeric(cleaned_df['rt'], errors='coerce')

    sorted_df = cleaned_df.sort_values(by='timestamp')

    output_dir = os.getenv('OUTPUT_DIR')
    output_filepath = os.path.join(output_dir, f'CallGraph_{file_id}.parquet')
    sorted_df.to_parquet(output_filepath)
