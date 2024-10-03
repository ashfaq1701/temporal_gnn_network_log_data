import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.forecasting.informer.data.dataset import WorkloadPredictionDataset
from src.forecasting.informer.models.model import Informer
from src.forecasting.informer.utils.metrics import metric
from src.forecasting.workload_prediction import build_workload_scaler, build_embedding_scaler, get_scaled_workloads, \
    get_scaled_embeddings, process_one_batch
from src.preprocess.functions import get_filtered_nodes_count


def test_workload_pred_noisy_data(args):
    embedding_scaler = build_embedding_scaler(args.embedding_scaling_type, args.embedding_scaling_factor)

    test_minutes = args.test_days * 24 * 60
    if args.reverse_test_data:
        test_start = 0
        test_end = test_minutes
    else:
        test_start = -test_minutes
        test_end = args.total_days * 24 * 60

    node_count = get_filtered_nodes_count()

    workloads = get_workloads(args.test_microservice_id, node_count)

    test_embeddings = None
    if args.use_temporal_embedding:
        test_embeddings = get_scaled_embeddings(test_start, test_end, embedding_scaler, args.test_microservice_id, node_count)

    if torch.cuda.is_available():
        device_string = 'cuda:{}'.format(args.gpu)
    else:
        device_string = 'cpu'
    device = torch.device(device_string)

    model = build_model(
        args.model_path,
        1 if args.test_microservice_id is not None else node_count,
        0 if not args.use_temporal_embedding else test_embeddings.shape[-1],
        device,
        args
    ).to(device)

    for seed in args.seeds:
        path_for_seed = os.path.join(args.output_dir, f'seed_{seed}')

        noisy_data_versions = get_noisy_test_versions(
            workloads, args.num_noisy_iters, 0, 1, seed
        )

        for modulation_factor, noisy_data in noisy_data_versions.items():
            workload_scaler = build_workload_scaler(args.scale_workloads_per_feature)
            scaled_noisy_data = scale_and_select_workloads(noisy_data, test_start, test_end, workload_scaler)

            current_noisy_ds = WorkloadPredictionDataset(
                workloads=scaled_noisy_data,
                embeddings=test_embeddings,
                start_minute=test_start,
                end_minute=test_end,
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.pred_len,
                workload_scaler=workload_scaler
            )

            metrics, preds, trues = predict_workloads(model, current_noisy_ds, device, args)

            path_for_mod_factor = os.path.join(path_for_seed, f'mod_factor_{modulation_factor}')
            os.makedirs(path_for_mod_factor, exist_ok=True)

            np.save(os.path.join(path_for_mod_factor, 'metrics.npy'), metrics)
            np.save(os.path.join(path_for_mod_factor, 'preds.npy'), preds)
            np.save(os.path.join(path_for_mod_factor, 'trues.npy'), trues)


def get_noisy_test_versions(data, num_noisy_iters, start_modulation_factor, end_modulation_factor, seed):
    np.random.seed(seed)
    max_value = np.max(data)
    random_noise = np.random.normal(0, 1, len(data))
    shifted_noise = random_noise - np.min(random_noise)
    normalized_noise = shifted_noise / np.max(shifted_noise)
    scaled_noise = normalized_noise * max_value

    modulation_increase = (end_modulation_factor - start_modulation_factor) / (num_noisy_iters - 1)

    noisy_datasets = {}

    for i in range(num_noisy_iters):
        current_modulation_factor = start_modulation_factor + i * modulation_increase
        all_noisy_workloads = []

        for j in range(data.shape[-1]):
            workloads_for_microservice = data[:, j]
            noisy_workloads = ((1.0 - current_modulation_factor) * scaled_noise
                               + current_modulation_factor * workloads_for_microservice)
            noisy_workloads[noisy_workloads < 0] = 0
            all_noisy_workloads.append(noisy_workloads)

        noisy_datasets[current_modulation_factor] = np.transpose(np.array(all_noisy_workloads))

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


def predict_workloads(model, test_ds, device, args):
    data_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )

    model.eval()

    preds = []
    trues = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        pred, true = process_one_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark, model, args, device
        )
        preds.append(pred.detach().cpu().numpy())
        trues.append(true.detach().cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    mae, mse, rmse, mape, mspe = metric(preds, trues)

    return np.array([mae, mse, rmse, mape, mspe]), preds, trues


def build_model(model_path, n_nodes, embedding_width, device, args):
    n_features = n_nodes + n_nodes * embedding_width
    n_labels = n_nodes

    model = Informer(
        n_features,
        n_labels,
        n_labels,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.factor,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.dropout,
        args.attn,
        args.embed,
        'm',
        args.activation,
        args.output_attention,
        args.distil,
        args.mix,
        device
    ).float()

    if device.type == 'cuda':
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model
