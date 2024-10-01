import os
import pickle

import numpy as np

from src.forecasting.informer.data.dataset import WorkloadPredictionDataset
from src.forecasting.workload_prediction import build_workload_scaler, build_embedding_scaler, get_scaled_workloads, \
    get_scaled_embeddings
from src.preprocess.functions import get_filtered_nodes_count


def test_workload_pred_noisy_data(
        model_path, seq_len, label_len, pred_len, test_node_id, test_days,
        use_temporal_embedding, reverse_test_data, num_datasets, seeds,
        scale_workloads_per_feature, embedding_scaling_type, embedding_scaling_factor,
        total_days
):
    embedding_scaler = build_embedding_scaler(embedding_scaling_type, embedding_scaling_factor)

    test_minutes = test_days * 24 * 60
    if reverse_test_data:
        test_start = 0
        test_end = test_minutes
    else:
        test_start = -test_minutes
        test_end = total_days * 24 * 60

    node_count = get_filtered_nodes_count()

    workloads = get_workloads(test_node_id, node_count)

    test_embeddings = None
    if use_temporal_embedding:
        test_embeddings = get_scaled_embeddings(test_start, test_end, embedding_scaler, test_node_id, node_count)

    for seed in seeds:
        noisy_data_versions = get_noisy_test_versions(
            workloads, num_datasets, 0, 1, seed
        )

        for noisy_data in noisy_data_versions:
            workload_scaler = build_workload_scaler(scale_workloads_per_feature)
            scaled_noisy_data = scale_and_select_workloads(noisy_data, test_start, test_end, workload_scaler)

            current_noisy_ds = WorkloadPredictionDataset(
                workloads=scaled_noisy_data,
                embeddings=test_embeddings,
                start_minute=test_start,
                end_minute=test_end,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                workload_scaler=workload_scaler
            )

            preds, metrics = predict_workloads(model_path, current_noisy_ds)


def get_noisy_test_versions(data, num_datasets, start_modulation_factor, end_modulation_factor, seed):
    max_value = np.max(data)
    random_noise = np.random.normal(0, 1, len(data))
    shifted_noise = random_noise - np.min(random_noise)
    normalized_noise = shifted_noise / np.max(shifted_noise)
    scaled_noise = normalized_noise * max_value

    modulation_decrease = (end_modulation_factor - start_modulation_factor) / (num_datasets - 1)

    noisy_datasets = []

    for i in range(num_datasets):
        current_modulation_factor = start_modulation_factor + i * modulation_decrease
        all_noisy_workloads = []

        for j in range(data.shape[-1]):
            workloads_for_microservice = data[:, j]
            noisy_workloads = ((1.0 - current_modulation_factor) * scaled_noise
                               + current_modulation_factor * workloads_for_microservice)
            noisy_workloads[noisy_workloads < 0] = 0
            all_noisy_workloads.append(noisy_workloads)

        noisy_datasets.append(np.transpose(np.array(all_noisy_workloads)))

    return noisy_datasets


def get_workloads(node_id, node_count):
    embedding_dir = os.getenv('EMBEDDING_DIR')

    with open(os.path.join(embedding_dir, 'workloads_over_time.pickle'), 'rb') as f:
        workloads = pickle.load(f)
        workloads = np.array(workloads)

    node_ids = [node_id] if node_id is not None else list(range(node_count))
    selected_workloads = workloads[:, node_ids]

    return selected_workloads


def scale_and_select_workloads(workloads, start_minute, end_minute, workload_scaler):
    scaled_workloads = workload_scaler.fit_transform(workloads)
    return scaled_workloads[start_minute:end_minute, :]


def predict_workloads(model_path, test_ds):
    return None, None
