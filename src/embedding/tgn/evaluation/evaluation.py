import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, \
    mean_absolute_error, mean_squared_error, r2_score

from src.embedding.tgn.utils.utils import get_unique_latest_nodes_with_indices, get_future_workloads, \
    combine_predictions, get_past_workloads


def eval_edge_prediction(model, negative_edge_sampler, valid_dataset, n_neighbors):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()

        for batch in valid_dataset:
            sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, edge_features_batch, _ = batch

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                  negative_samples, timestamps_batch,
                                                                  edge_features_batch, n_neighbors)

            valid_dataset.add_batch_to_neighbor_finder(batch[:-1])

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)


def eval_workload_prediction(
        tgn,
        workload_predictor,
        valid_dataset,
        embedding_buffer,
        workloads,
        device,
        loss_criterion,
        start_minute,
        n_nodes,
        n_past,
        n_future,
        n_neighbors
):
    true_workloads = np.empty((0, 3))
    pred_workloads = np.empty((0, 3))
    nodes = np.array([])
    timestamps = np.array([])

    with torch.no_grad():
        tgn.eval()
        workload_predictor.eval()

        current_minute = start_minute
        loss = 0

        for batch in valid_dataset:
            sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, \
                edge_features_batch, current_file_end = batch

            with torch.no_grad():
                _, destination_embeddings, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                               destinations_batch,
                                                                               destinations_batch,
                                                                               timestamps_batch,
                                                                               edge_idxs_batch,
                                                                               n_neighbors)
                destination_embeddings_np = destination_embeddings.detach().cpu().numpy()

            valid_dataset.add_batch_to_neighbor_finder(batch[:-1])

            nodes_with_latest_indices = get_unique_latest_nodes_with_indices(destinations_batch, timestamps_batch)
            latest_destination_embeddings = destination_embeddings_np[nodes_with_latest_indices[:, 1]]
            embedding_buffer.add_embeddings(nodes_with_latest_indices[:, 0], latest_destination_embeddings)

            if current_file_end:
                past_workloads = get_past_workloads(
                    np.arange(0, n_nodes),
                    current_minute,
                    workloads,
                    n_past
                )
                future_workloads = get_future_workloads(
                    np.arange(0, n_nodes),
                    current_minute,
                    workloads,
                    n_future
                )
                embedding_buffer.add_to_buffer(current_minute, past_workloads, future_workloads)

                if embedding_buffer.is_buffer_full():
                    embeddings_batch, nodes_batch, past_workloads_batch, future_workloads_batch, timesteps_batch \
                        = embedding_buffer.get_batch()
                    preds = workload_predictor.predict_workload(embeddings_batch, nodes_batch)

                    workloads_batch_tensor = torch.tensor(future_workloads_batch, dtype=torch.float32).to(device)
                    workload_predictor_loss = loss_criterion(preds, workloads_batch_tensor)
                    loss += workload_predictor_loss.item()

                    preds_np = preds.cpu().detach().numpy()
                    true_workloads = np.concatenate((true_workloads, future_workloads_batch), axis=0)
                    pred_workloads = np.concatenate((pred_workloads, preds_np), axis=0)

                    nodes = np.concatenate((nodes, nodes_batch))
                    timestamps = np.concatenate((timestamps, timesteps_batch))

                current_minute += 1

        embeddings_batch, nodes_batch, past_workloads_batch, future_workloads_batch, timesteps_batch \
            = embedding_buffer.get_batch()
        if embeddings_batch.shape[0] > 0:
            preds = workload_predictor.predict_workload(embeddings_batch, nodes_batch)

            workloads_batch_tensor = torch.tensor(future_workloads_batch, dtype=torch.float32).to(device)
            workload_predictor_loss = loss_criterion(preds, workloads_batch_tensor)
            loss += workload_predictor_loss.item()

            preds_np = preds.cpu().detach().numpy()
            true_workloads = np.concatenate((true_workloads, future_workloads_batch), axis=0)
            pred_workloads = np.concatenate((pred_workloads, preds_np), axis=0)

            nodes = np.concatenate((nodes, nodes_batch))
            timestamps = np.concatenate((timestamps, timesteps_batch))

    mae = mean_absolute_error(true_workloads, pred_workloads)
    mse = mean_squared_error(true_workloads, pred_workloads)
    r2 = r2_score(true_workloads, pred_workloads)
    avg_loss = loss / valid_dataset.get_total_batches()

    combined_predictions = combine_predictions(true_workloads, pred_workloads, nodes, timestamps)

    return mae, mse, r2, avg_loss, combined_predictions
