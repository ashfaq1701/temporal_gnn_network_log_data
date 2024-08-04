import numpy as np
from scipy.fft import fft, fftfreq

from src.preprocess.functions import get_node_label_encoder, get_downstream_counts_object, get_microservice_workload


def get_seasonality(data):
    data_np = np.array(data)
    N = len(data_np)
    T = 1.0  # Sample spacing (1 minute)

    # Compute the FFT
    yf = fft(data)
    xf = fftfreq(N, T)[:N // 2]

    # Frequency corresponding to daily cycle
    daily_freq = 1 / 1440.0

    # Find the index of the frequency closest to daily_freq
    idx = np.argmin(np.abs(xf - daily_freq))

    # Magnitude at the daily frequency
    daily_periodicity = np.abs(yf[idx])
    return daily_periodicity


def compute_seasonality_of_microservices():
    label_encoder = get_node_label_encoder()
    microservices = label_encoder.classes_
    downstream_counts = get_downstream_counts_object()
    all_seasonality = {}
    for microservice in microservices:
        workload = get_microservice_workload(downstream_counts, microservice)
        seasonality = get_seasonality(workload)
        all_seasonality[microservice] = seasonality
    return all_seasonality
