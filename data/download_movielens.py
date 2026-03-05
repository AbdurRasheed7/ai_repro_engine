import urllib.request
import zipfile
import os

def download_movielens():
    """Download MovieLens 100K dataset automatically"""
    
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "./data/ml-100k.zip"
    extract_path = "./data/"
    
    if os.path.exists("./data/ml-100k"):
        print("✅ MovieLens dataset already exists!")
        return "./data/ml-100k"
    
    os.makedirs("./data", exist_ok=True)
    
    print("📥 Downloading MovieLens 100K dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    os.remove(zip_path)
    print("✅ MovieLens dataset ready at ./data/ml-100k!")
    return "./data/ml-100k"