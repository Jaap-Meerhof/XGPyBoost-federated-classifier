from customddsketch import DDSketch
import numpy as np
from tqdm import tqdm
import pickle
import os

def find_splits(X, EA:float, N_BINS:int):
    """create splits using DDSketch. also adds the minimun and maximum value as a split

    Args:
        X (_type_): _description_
        EA (float): error for DDSKetch algorithm
        N_BINS (int): _description_

    Returns:
        list[list[float]]: splits with shape (num_features, n_bins)
    """
    sketch:list[DDSketch] = []
    splits:list[list[float]] = []
    for feature in tqdm(range(X.shape[1]), desc="> Federated XGBoost DDSketch progress"):
        sketch.append(DDSketch(EA))

        my_max = np.max(X[:, feature])
        my_min = np.min(X[:, feature])
        for x in X[:,feature]:
            sketch[feature].add(x)
        splits.append([sketch[feature].get_quantile_value(i/N_BINS) for i in range(N_BINS)])
        splits[-1].insert(0, my_min)
        splits[-1].append(my_max)
    return splits

def data_to_histo(X, num_bins=255):
    splits: list[list[float]] = []
    for feature in range(X.shape[1]):
        range_min = np.min(X[:, feature])
        range_max = np.max(X[:, feature])
        
        bin_edges = np.linspace(range_min, range_max, num=num_bins-1)
        # bin_indices = np.digitize(X[:, feature], bin_edges) -1
        # bin_counts = np.bincount(bin_indices, minlength=num_bins)
        # plt.bar(range(num_bins), bin_counts, align='center')
        # plt.xticks(range(num_bins))
        # plt.xlabel('Bins')
        # plt.ylabel('Frequency')
        # plt.title('Histogram')
        # plt.show()
        splits.append(bin_edges)
    return splits

def saveVar(var:any, name:str): # .kl
    name = name +"pkl"
    with open(name, 'wb') as file:
        pickle.dump(var)

def getVar(name:str):
    name = name + ".kl"
    if os.path.exists(name):
        with open(name, 'rb') as file:
            splits = pickle.load(file)