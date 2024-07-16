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
