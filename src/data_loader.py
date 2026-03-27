import os
import urllib.request
import zipfile

def download_movielens_100k(dest_dir="data/raw"):
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "ml-100k.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading {url} to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
    
    extract_dir = os.path.join(dest_dir, "ml-100k")
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
            
    print("Download and extraction complete.")

if __name__ == "__main__":
    download_movielens_100k()
