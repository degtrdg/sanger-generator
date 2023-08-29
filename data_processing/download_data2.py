import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import concurrent.futures
import threading
from tqdm import tqdm

base_url = "https://ddbj.nig.ac.jp/public/ddbj_database/dta/"
save_path = './data'
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)  # Adjust as necessary

pbar = tqdm(total=0)

def download_file(url, path):
    print(f"Starting download: {path}")
    file_response = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in file_response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    pbar.update(1)
    print(f"Download completed: {path}")

def download_recursive(url, path):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for link in soup.select('td > a')[1:]:  # skip the first (parent) link
        href = link.get('href')
        href_url = urljoin(url, href)

        # If link ends with '/', it's a directory. Recurse into it
        if href.endswith('/'):
            new_path = os.path.join(path, href)
            os.makedirs(new_path, exist_ok=True)
            download_recursive(href_url, new_path)
        else:  # It's a file, download it
            local_filename = os.path.join(path, href)

            # Skip download if file exists
            if os.path.exists(local_filename):
                print(f"File {local_filename} already exists. Skipping.")
            else:
                executor.submit(download_file, href_url, local_filename)
                pbar.total += 1

download_recursive(base_url, save_path)
executor.shutdown(wait=True)  # Wait for all downloads to finish
pbar.close()
