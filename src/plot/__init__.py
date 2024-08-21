import os
import matplotlib.pyplot as plt
import numpy as np


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


from src.utils import get_test_workloads, get_target_microservice_id


def get_pred_and_true_workloads(filepath, take_first=True):
    test_workloads = get_test_workloads()
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


def plot_pred_and_true_workloads(filepath, title, save_filepath, take_first=True):
    pred_workloads, true_workloads = get_pred_and_true_workloads(filepath, take_first)

    train_days = int(os.getenv('WORKLOAD_PREDICTION_TRAINING_DAYS'))
    valid_days = int(os.getenv('WORKLOAD_PREDICTION_VALIDATION_DAYS'))
    test_days = int(os.getenv('WORKLOAD_PREDICTION_TEST_DAYS'))

    test_workloads = get_test_workloads()
    start_idx, end_idx = len(test_workloads) - len(pred_workloads), len(test_workloads)

    x_values = np.arange(0, len(test_workloads))
    x_plot = x_values[start_idx:end_idx]

    minutes_per_day = 1440
    days = np.arange(0, test_days + 1)
    day_labels = [str(day + train_days + valid_days) for day in days]

    plt.figure(figsize=(14, 7))

    plt.plot(x_plot, true_workloads, label='True Workloads', color='red')
    plt.plot(x_plot, pred_workloads, label='Predicted Workloads', color='blue', linestyle='--')

    plt.xlabel('Day')
    plt.ylabel('Workload (Calls Per Minute)')
    plt.xticks(days * minutes_per_day, day_labels)
    plt.title(title)
    plt.legend()
    plt.savefig(save_filepath)
