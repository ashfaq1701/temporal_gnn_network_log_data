import tarfile

import requests
from retrying import retry
import os
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
