import os
import pandas as pd
import pickle
from src.preprocess.functions import get_stats_object, get_all_microservices
from sklearn.preprocessing import LabelEncoder


def produce_final_format_data():
    stats = get_stats_object()
    all_nodes = get_all_microservices(stats)
    label_encoder = get_label_encoder(all_nodes)

    input_dir = os.getenv('PER_MINUTE_OUTPUT_DIR')
    output_dir = os.getenv('FINAL_DATA_DIR')

    running_total = 0

    for idx in range(20160):
        filepath = os.path.join(input_dir, f'CallGraph_{idx}.parquet')
        df = pd.read_parquet(filepath)

        transformed_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'um': label_encoder.transform(df['um']),
            'dm': label_encoder.transform(df['dm']),
        })

        new_index = pd.RangeIndex(start=running_total, stop=running_total + len(df))
        transformed_df.index = new_index

        running_total += len(df)

        output_filepath = os.path.join(output_dir, f'data_{idx}.parquet')
        transformed_df.to_parquet(output_filepath)
        print(f'Stored file {output_filepath}')

    metadata_path = os.getenv('METADATA_DIR')
    label_encoder_file_path = os.path.join(metadata_path, 'label_encoder.pickle')
    with open(label_encoder_file_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f'Stored label encoder in {label_encoder_file_path}')


def get_label_encoder(nodes):
    label_encoder = LabelEncoder()
    label_encoder.fit(nodes)
    return label_encoder
