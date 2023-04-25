import numpy as np

class Histogram:

    def __init__(self):
        pass

    def create_histogram(self, X, n_bins):
        features = X.shape[1]
        hist = np.zeros((features, n_bins), dtype=np.int32)
        for j in range(features):
            feat_vals, feat_idx = np.unique(X[:, j], return_inverse=True)
            # Compute histogram of gradients for each unique feature value
            for k in range(len(feat_vals)):
                idx = feat_idx == k
                hist[j, k] = np.sum(y[idx])

            for k in range(1, len(feat_vals)):
                hist[j, k] += hist[j, k-1]
                hist[j, :-1] = hist[j, 1:] - hist[j, :-1] 
        return hist

Histy = Histogram()
X = np.array([[1,2,3,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,7,8,9,9,9,9,9,10],[1,2,3,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,7,8,9,9,9,9,9,10]])
myhist = Histy.create_histogram(X, 10)
pass