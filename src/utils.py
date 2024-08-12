import tarfile

import requests
from retrying import retry
import os
import numpy as np
from urllib.parse import urlparse


def get_filename_from_url(url):
    """
    Extracts the filename from a URL.

    :param url: The URL of the file.
    :return: The filename extracted from the URL.
    """
    parsed_url = urlparse(url)
    return parsed_url.path.split('/')[-1]


@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def download_file(url, dest_path):
    try:
        filename = get_filename_from_url(url)
        filepath = os.path.join(dest_path, filename)

        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Check if the request was successful
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
                    file.write(chunk)

        print(f"Downloaded file successfully: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        raise


def if_file_exists(dest_path, filename):
    filepath = os.path.join(dest_path, filename)
    return os.path.exists(filepath)


def extract_tar_gz(filepath, extract_to):
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_to)


def count_trimmed_values(df, col_name):
    df[f'trimmed_{col_name}'] = df[col_name].str.strip()
    value_counts = df[f'trimmed_{col_name}'].value_counts()
    return value_counts.to_dict()


def get_files_in_directory_with_ext(directory, extension):
    files = os.listdir(directory)
    pickle_files = [file for file in files if file.endswith(extension)]
    return pickle_files


def combine_means_and_stds(means, stds, sizes):
    # Calculate the combined mean
    total_size = np.sum(sizes)
    combined_mean = np.sum(np.array(means) * np.array(sizes)) / total_size

    # Calculate the combined variance
    sum_of_squares = np.sum((np.array(sizes) - 1) * (np.array(stds) ** 2))
    sum_of_square_of_diff = np.sum(np.array(sizes) * (np.array(means) - combined_mean) ** 2)
    combined_variance = (sum_of_squares + sum_of_square_of_diff) / (total_size - 1)

    # Calculate the combined standard deviation
    combined_std = np.sqrt(combined_variance)

    return combined_mean, combined_std


def get_training_and_validation_file_indices(training_days, validation_days):
    training_minutes = int(training_days * 24 * 60)
    validation_minutes = int(validation_days * 24 * 60)
    return (0, training_minutes), (training_minutes, training_minutes + validation_minutes)

