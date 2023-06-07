import numpy as np
import copy
from params import Params
import multiprocessing
from threading import Thread
import concurrent.futures

class TreeNode:
    def __init__(self, instances, depth = 0):
        self.lchild = None
        self.rchild = None
        self.feature = None
        self.threshold = None
        self.is_leaf = False
        self.weight = None
        self.instances = instances
        self.depth = depth

    # from lgbm c++ source
    def threshold_l1(self, w, alpha):
        reg_s = np.max((0.0, np.abs(w) - alpha))
        return np.sign(w)*reg_s

    def calc_weight(self, G, H, params):
        w = -self.threshold_l1(G, params.alpha) / (H + params.lam) # square root of l1?

        if params.max_delta_step != 0:
            w = np.clip(w, -params.max_delta_step, params.max_delta_step)

        return w

    def calc_gain(self, G, H, params):
        if H < params.min_child_weight:
            return 0.0
        w = self.calc_weight(G, H, params)

        # this is 2* the terms from equation 7 to avoid the slower 1/2 division (copying from c++ source)
        gain = -(2*G*w + (H+params.lam)*(w**2))
        return gain - 2*params.alpha * abs(w)

    def calc_split_gain(self, G, H, G_l, G_r, H_l, H_r, params):
        gain_left = self.calc_gain(G_l, H_l, params)
        gain_right = self.calc_gain(G_r, H_r, params)
        gain_root = self.calc_gain(G, H, params)

        gain = gain_left + gain_right - gain_root

        # make gain 0 (so don't split) if either child violates min_child_weight condition
        if gain < params.gamma or H_l < params.min_child_weight or H_r < params.min_child_weight:
            return 0.0
        return gain

    def _get_leaf_weight(self, G, H, params):
        return self.calc_weight(G, H, params) * params.eta

    def _get_child_instances(self, X):
        linstances = np.logical_and(X[:, self.feature] < self.threshold, self.instances)
        rinstances = np.logical_and(X[:, self.feature] >= self.threshold, self.instances)
        return linstances, rinstances



    # exact greedy algorithm for enumerating all possibe splits (algorithm 1)
    def enumerate_splits(self, X, grad, hess, params):
        if self.depth >= params.max_depth:
            # if root, don't check min_child_weight
            if self.depth == 0:
                params = copy.deepcopy(params)
                params.min_child_weight = 0
            self.is_leaf = True
            self.weight = self._get_leaf_weight(np.sum(grad[self.instances]), np.sum(hess[self.instances]), params)
            return

        # subset X, grad, and hess to node's instance set
        X_i = X[self.instances, :]
        grad_i = grad[self.instances]
        hess_i = hess[self.instances]
        G = np.sum(grad_i)
        H = np.sum(hess_i)
        best_gain = 0.0

        # loop over feature columns

        # def find_highest_gain(feature):
        #     best_gain_feature = 0
        #     threshold_feature = None
        #     G_l, H_l, = 0, 0
        #     # can speed this up significantly by pre-sorting columns
        #     subset_ordered_idx = X_i[:, feature].argsort()
        #     feature_ordered_subset = X_i[subset_ordered_idx, feature]
        #     grad_ordered_subset = grad_i[subset_ordered_idx]
        #     hess_ordered_subset = hess_i[subset_ordered_idx]

        #     for i in range(feature_ordered_subset.shape[0]-1): #TODO Parralise
        #         G_l += grad_ordered_subset[i]
        #         H_l += hess_ordered_subset[i]
        #         G_r = G - G_l
        #         H_r = H - H_l

        #         # don't consider split if it doesn't uniquely separate obs (i.e. ignore repeat values)
        #         if feature_ordered_subset[i] == feature_ordered_subset[i+1]:
        #             continue

        #         gain = self.calc_split_gain(G, H, G_l, G_r, H_l, H_r, params)
        #         if gain >= best_gain_feature:
        #             best_gain_feature = gain
        #             # use midpoint of between feature values as threshold
        #             threshold_feature = (feature_ordered_subset[i] + feature_ordered_subset[i + 1]) / 2
        #     return best_gain_feature, threshold_feature

        # num_cores = multiprocessing.cpu_count()
        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:

        #     futures = [executor.submit(find_highest_gain, feature) for feature in range(X.shape[1])]

        #     concurrent.futures.wait(futures)
        #     result = [future.result() for future in futures]
        #     for feature in range(X.shape[1]):
        #         gain = result[feature][0]
        #         if gain >= best_gain:
        #             best_gain = gain
        #             self.feature = feature
        #             self.threshold = result[feature][1]
        #     pass

        for feature in range(X.shape[1]): # TODO parralerisation
            G_l, H_l, = 0, 0
            # can speed this up significantly by pre-sorting columns
            subset_ordered_idx = X_i[:, feature].argsort()
            feature_ordered_subset = X_i[subset_ordered_idx, feature]
            grad_ordered_subset = grad_i[subset_ordered_idx]
            hess_ordered_subset = hess_i[subset_ordered_idx]

            for i in range(feature_ordered_subset.shape[0]-1): #TODO Parralise
                G_l += grad_ordered_subset[i]
                H_l += hess_ordered_subset[i]
                G_r = G - G_l
                H_r = H - H_l

                # don't consider split if it doesn't uniquely separate obs (i.e. ignore repeat values)
                if feature_ordered_subset[i] == feature_ordered_subset[i+1]:
                    continue

                gain = self.calc_split_gain(G, H, G_l, G_r, H_l, H_r, params)
                if gain >= best_gain:
                    best_gain = gain
                    self.feature = feature
                    # use midpoint of between feature values as threshold
                    self.threshold = (feature_ordered_subset[i] + feature_ordered_subset[i + 1]) / 2

        # means there's no further gain
        if best_gain == 0:
            # if root, don't check min_child_weight
            if self.depth == 0:
                params = copy.deepcopy(params)
                params.min_child_weight = 0
            self.is_leaf = True
            self.weight = self._get_leaf_weight(np.sum(grad_i), np.sum(hess_i), params)
            return

        linstances, rinstances = self._get_child_instances(X)
        self.lchild, self.rchild = TreeNode(linstances, depth = self.depth + 1), TreeNode(rinstances, depth = self.depth + 1)

    def predict_one(self, x):
            if self.is_leaf:
                return self.weight
            else:
                if x[self.feature] < self.threshold:
                    return self.lchild.predict_one(x)
                else:
                    return self.rchild.predict_one(x)

    # this is gonna be reallyy slow without threading
    def predict(self, X):
        preds = np.apply_along_axis(self.predict_one, 1, X)
        # probabilities = 1.0 / (1.0 + np.exp(-preds))
        # binary_predictions = np.where(probabilities >= 0.5, 1, 0)
        return preds