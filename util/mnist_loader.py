import pickle
import numpy as np
from sklearn.cluster import KMeans

def load_mnist(pickle_path):
    with open(pickle_path, 'rb') as f:
        x_train, x_test, y_train, y_test = pickle.load(f).values()
    return x_train, x_test, y_train, y_test


def load_mnist_semisupervised(pickle_path, 
                              N_labeled=1, 
                              N_unlabeled=0, 
                              seed=1234):
    x_train, x_test, y_train, y_test = load_mnist(pickle_path)
    
    labels = list(set(y_train))
    print(f"Labels: {labels}")
    
    indexes_labeled = []
    indexes_unlabaled = []
    
    np.random.seed(seed)
    
    for lab in labels:
        lab_indexes = np.where(y_train==lab)[0]
        if N_unlabeled==0:
            sampled_indexes = np.random.choice(lab_indexes,
                                               size=N_labeled)
            indexes_labeled.extend(sampled_indexes)
            indexes_unlabaled.extend(np.delete(np.arange(len(lab_indexes)), sampled_indexes))
        else:
            sampled_indexes = np.random.choice(np.where(y_train==lab)[0], 
                                               size=N_labeled + N_unlabeled)
            indexes_labeled.extend(sampled_indexes[:N_labeled])
            indexes_unlabaled.extend(sampled_indexes[N_labeled:])

    
    data_dict = {}
    
    data_dict["x_train_labeled"] = x_train[indexes_labeled,:]
    data_dict["x_train_unlabeled"] = x_train[indexes_unlabaled,:]

    data_dict["y_train_labeled"] = y_train[indexes_labeled]
    data_dict["y_train_unlabeled"] = y_train[indexes_unlabaled]
    
    data_dict["x_test"] = x_test
    data_dict["y_test"] = y_test
    
    print(f"Labeled Train Shape   : {data_dict['x_train_labeled'].shape}")
    print(f"UnLabeled Train Shape : {data_dict['x_train_unlabeled'].shape}")

    return data_dict
    