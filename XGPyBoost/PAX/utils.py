from customddsketch import DDSketch
import numpy as np
from tqdm import tqdm


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