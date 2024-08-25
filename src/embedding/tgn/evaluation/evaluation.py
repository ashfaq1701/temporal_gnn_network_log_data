import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


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
