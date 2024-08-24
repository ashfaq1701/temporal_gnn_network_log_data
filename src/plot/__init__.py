import os
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_test_workloads, get_target_microservice_id, get_train_workloads, get_valid_workloads


def plot_microservice_workload(microservice, workloads):
    time_points = range(len(workloads))
    plt.plot(time_points, workloads, label='Workload')

    plt.xlabel('Days')
    plt.ylabel('Workload')
    plt.title(f'Workloads Over 14 Days - {microservice}')

    num_days = 14
    minutes_per_day = 1440
    days = np.arange(0, num_days + 1)
    day_labels = [str(day) for day in days]

    plt.xticks(days * minutes_per_day, day_labels)
    plt.show()


def get_pred_and_true_workloads(filepath, take_first=True, microservice_id=None):
    test_workloads = get_test_workloads(microservice_id)

    if microservice_id is not None:
        target_microservice_id = microservice_id
    else:
        target_microservice_id = get_target_microservice_id()

    preds = np.load(filepath)
    pred_len = preds.shape[0] + preds.shape[1] - 1
    selected_test_workloads = test_workloads[-pred_len:]

    pred_workloads = np.zeros((pred_len,), dtype=np.float32)

    for idx in range(preds.shape[0]):
        pred_group = preds[idx, :, :]
        n_future = pred_group.shape[0]
        pred_n_nodes = pred_group.shape[1]

        node_id = 0 if pred_n_nodes == 1 else target_microservice_id

        if take_first:
            if idx < preds.shape[0] - 1:
                pred_workloads[idx] = pred_group[0, node_id]
            else:
                pred_workloads[idx:idx + n_future] = pred_group[:, node_id]
        else:
            if idx == 0:
                pred_workloads[idx:idx + n_future] = pred_group[:, node_id]
            else:
                pred_workloads[idx + n_future - 1] = pred_group[-1, node_id]

    return pred_workloads, selected_test_workloads


def plot_pred_and_true_workloads(filepath, title, save_filepath, take_first=True, microservice_id=None):
    pred_workloads, true_workloads = get_pred_and_true_workloads(filepath, take_first, microservice_id)

    train_days = int(os.getenv('WORKLOAD_PREDICTION_TRAINING_DAYS'))
    valid_days = int(os.getenv('WORKLOAD_PREDICTION_VALIDATION_DAYS'))
    test_days = int(os.getenv('WORKLOAD_PREDICTION_TEST_DAYS'))

    test_workloads = get_test_workloads(microservice_id)
    start_idx, end_idx = len(test_workloads) - len(pred_workloads), len(test_workloads)

    x_values = np.arange(0, len(test_workloads))
    x_plot = x_values[start_idx:end_idx]

    minutes_per_day = 1440
    days = np.arange(0, test_days + 1)
    day_labels = [str(day + train_days + valid_days) for day in days]

    plt.figure(figsize=(14, 7))

    plt.plot(x_plot, true_workloads, label='True Workloads', color='teal')
    plt.plot(x_plot, pred_workloads, label='Predicted Workloads', color='coral', linestyle='--')

    plt.xlabel('Day', fontsize=12, fontweight='bold')
    plt.ylabel('CPM', fontsize=12, fontweight='bold')
    plt.xticks(days * minutes_per_day, day_labels)
    plt.title(title, fontsize=14, fontweight='bold')

    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(save_filepath, bbox_inches='tight')



def plot_full_workloads(filepath, title, save_filepath, take_first=True, microservice_id=None):
    pred_workloads, true_workloads = get_pred_and_true_workloads(filepath, take_first, microservice_id)
    train_workloads = get_train_workloads(microservice_id)
    valid_workloads = get_valid_workloads(microservice_id)
    test_workloads = get_test_workloads(microservice_id)

    minutes_per_day = 1440
    days = np.arange(0, 15)
    day_labels = [str(day) for day in np.arange(0, 15)]

    plt.figure(figsize=(14, 7))

    plt.plot(range(len(train_workloads)), train_workloads, label="Train Workloads", color="green", linestyle='-')
    plt.plot(range(len(train_workloads), len(train_workloads) + len(valid_workloads)), valid_workloads,
             label="Valid Workloads", color="purple", linestyle='-')
    plt.plot(range(len(train_workloads) + len(valid_workloads), 20160), test_workloads, label='Test Workloads',
             color='teal', linestyle='-')
    plt.plot(range(20160 - len(pred_workloads), 20160), pred_workloads, label='Predicted Workloads', color='coral',
             linestyle='--')

    plt.xlabel('Day', fontsize=12, fontweight='bold')
    plt.ylabel('CPM', fontsize=12, fontweight='bold')
    plt.xticks(days * minutes_per_day, day_labels)
    plt.title(title, fontsize=14, fontweight='bold')

    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(save_filepath, bbox_inches='tight')

