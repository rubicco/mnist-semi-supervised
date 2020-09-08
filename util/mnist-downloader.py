# Source: (edited slightly)
# https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py
import os
import gzip
import pickle
from urllib import request
import numpy as np

DATASET_DIR = "./data/"
DOWNLOAD_DIR = "./data/tmp/"

BASE_URL = "http://yann.lecun.com/exdb/mnist/"

TARGET_LIST = [
    ("train_images", "train-images-idx3-ubyte.gz"),
    ("test_images", "t10k-images-idx3-ubyte.gz"),
    ("train_labels", "train-labels-idx1-ubyte.gz"),
    ("test_labels", "t10k-labels-idx1-ubyte.gz")
]


def download():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        
    for _, file_name in TARGET_LIST:
        print(f"Downloading {DOWNLOAD_DIR + file_name}...")
        request.urlretrieve(BASE_URL + file_name, DOWNLOAD_DIR + file_name)
    print("Download Completed!")
    

def save_as_picke():
    mnist = {}
    
    for file_tag, file_name in TARGET_LIST[:2]:
        with gzip.open(DOWNLOAD_DIR + file_name, 'rb') as f:
            mnist[file_tag] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)/255
            
    for file_tag, file_name in TARGET_LIST[-2:]:
        with gzip.open(DOWNLOAD_DIR + file_name, 'rb') as f:
            mnist[file_tag] = np.frombuffer(f.read(), np.uint8, offset=8)
            
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
            
    with open(DATASET_DIR + "mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print(f"Saving as pickle complete: {DATASET_DIR + 'mnist.pkl'}")
    
def clean():
    if os.path.exists(DOWNLOAD_DIR):
        for _, file_name in TARGET_LIST:
            if os.path.exists(DOWNLOAD_DIR + file_name):
                os.remove(DOWNLOAD_DIR + file_name)
        os.rmdir(DOWNLOAD_DIR)
        print("Download directory (./tmp) is cleaned and removed.")
     
def main():
    download()
    save_as_picke()
    clean()
    
if __name__ == '__main__':
    main()