import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base_url = "https://ddbj.nig.ac.jp/public/ddbj_database/dta/"
save_path = './data'


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
            file_response = requests.get(href_url, stream=True)
            local_filename = os.path.join(path, href)

            with open(local_filename, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


download_recursive(base_url, save_path)
