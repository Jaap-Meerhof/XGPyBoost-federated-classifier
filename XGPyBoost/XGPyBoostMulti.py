from dataclasses import dataclass
import numpy as np
import pandas as pd
import copy
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder


@dataclass
class Params:
    #__slots__ = ['n_trees', 'max_depth', 'eta', 'lam', 'alpha', 'gamma', 'min_child_weight', 'max_delta_step']
    n_trees: int
    max_depth: int = 6
    eta: float = 0.3
    lam: float = 1.0
    alpha: float = 0.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    max_delta_step : float = 0.0


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
        w = -self.threshold_l1(G, params.alpha) / (H + params.lam)

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
        for feature in range(X.shape[1]):
            G_l, H_l, = 0, 0
            # can speed this up significantly by pre-sorting columns
            subset_ordered_idx = X_i[:, feature].argsort()
            feature_ordered_subset = X_i[subset_ordered_idx, feature]
            grad_ordered_subset = grad_i[subset_ordered_idx]
            hess_ordered_subset = hess_i[subset_ordered_idx]

            for i in range(feature_ordered_subset.shape[0]-1):
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

class XGPyBoostMulti:
    def __init__(
        self,
        n_trees,
        obj,
        **kwargs
    ):
        self.obj = obj
        self.params = Params(n_trees, **kwargs)

    def _get_init_preds(self, base_margin, X):
        if isinstance(base_margin, np.ndarray):
            preds = base_margin
        elif isinstance(base_margin, (int, float, complex)):
            preds = np.repeat(base_margin, X.shape[0])
        else:
            assert("Base margin must be number or np array")
        return preds.astype(np.float64)

    def _get_init_preds_y(self, y):
        '''using the frequency of labels in y create a good first bet! '''
        self.le = LabelEncoder()
        labels = self.le.fit_transform(y)
        Y = self._to_categorical(labels) # [X_size [n_classes]] labeled with integers 0 to n_classes-1 one-hot encoded
        del labels
        
        y_proba = np.full(Y.shape, 1/Y.shape[1]) # inpreitial probabilities
        return Y, y_proba

    def _fit_tree(self, X, grad, hess, params):
        # initialize only root node
        root = TreeNode(np.full(X.shape[0], True))#np.arange(X.shape[0]))
        stack = [root]
        # recursively add nodes to the stack until we're done
        while len(stack) > 0:
            node = stack.pop()
            node.enumerate_splits(X, grad, hess, params)
            if not node.is_leaf:
                # add both children to front of list
                stack[0:0] = [node.lchild, node.rchild]
            
        return root

    def fit(self, X, y, base_margin = 0):
        self.n_classes = np.amax(y) + 1
        Y, self.initial_preds = self._get_init_preds_y(y)
        preds = deepcopy(self.initial_preds) # TODO has to be of [nclasses]
        self.trees = [[] for i in range(self.n_classes)]
        for i in range(self.params.n_trees):
            print("starting with tree {}".format(i))
            grad, hess = self.obj(y, preds)
            for c in range(self.n_classes):
                # get initial grads and hess
                tree = self._fit_tree(X, grad[:, c], hess[:, c], self.params)
                self.trees[c].append(tree)

                # preds = self.predict(X)
                preds[:, c] = tree.predict(X)


    '''http://ethen8181.github.io/machine-learning/trees/gbm/gbm.html#Gradient-Boosting-Machine-(GBM) '''
    def _to_categorical(self, y):
        """one hot encode class vector y"""
        self.n_classes = np.amax(y) + 1
        Y = np.zeros((y.shape[0], self.n_classes))
        for i in range(y.shape[0]):
            Y[i, y[i]] = 1.0

        return Y


    def predict(self, X, base_margin = 0):
        

        probas = np.zeros((X.shape[0], self.n_classes, self.params.n_trees+1))
        
        y_pred = np.zeros((X.shape[0], self.n_classes, self.params.n_trees+1))
        
        y_pred[:, :, 0] = self.initial_preds[:X.shape[0], :]

        for c in range(self.n_classes):
            for i, tree in enumerate(self.trees[c]):
                y_pred[:, c, i+1] = tree.predict(X)

        

        # now y_preds are filled
        for i in range(self.params.n_trees+1):
            for rowid in range(y_pred.shape[0]):
                row = y_pred[rowid, : , i]
                wmax = max(row) # line 100 multiclass_obj.cu
                wsum =0.0
                for y in row : wsum +=  np.exp(y - wmax)   
                probas[rowid,:, i] = np.exp(row -wmax) / wsum
                
        probas = np.average(probas, axis=2)

        # tmp = [np.exp(x)/sum(np.exp(tmp)) for x in tmp]
        # binary = np.where(p >= 0.5, 1, 0)
        # binary_predictions[:, i+1] = binary
        # binary_predictions[:, i+1]
            
        # for i in range(X.shape[0]):
        #     average = sum(probas[i, :])/len(probas[i, :])
        #     probas[i, c] =  average

        # TODO take the highest probability and return its location in the list
        pass 
            # for i in range(X.shape[0]):
            #     votes[i][c] = np.argmax(np.bincount(binary_predictions[i, :]))

        highets_prob = np.zeros((X.shape[0], self.n_classes), dtype='int64')
        for i in range(X.shape[0]):
            highets_prob[i] = np.where(probas[i] == np.amax(probas[i]), 1, 0)

        return [ np.argmax(probdrow ) for probdrow in highets_prob ]
    
    def predict_proba(self, X, base_margin = 0):
        return 1/(1+np.exp(-self.predict(X, base_margin)))
        
def mse_obj(y_true, y_pred):
    grad = y_pred - y_true
    hess = np.ones_like(y_true)
    return grad, hess

def hessfunc(x):
    # return 0.0001
    # return max(2*x * (1.0 - x), 1e-6 )
    
    return max(2*x*(1-x), 1e-6)


def logistic_obj(y_true, y_pred):
    test = 1.0/(1.0+np.exp(-y_pred))
    grad = test - y_true
    hess = test * (1.0 - test)
    return grad, hess

def softprob_obj(y_true, y_pred, c):
    '''y_true = y, not one-hot-encoded just numbers '''
    # grad = np.zeros((y_pred.shape[0], y_pred.shape[1]), dtype=float) # for multi-class
    # hess = np.zeros((y_pred.shape[0], y_pred.shape[1]), dtype=float) # for multi-class
    grad = np.zeros((y_pred.shape[0]))
    hess = np.zeros((y_pred.shape[0]))
    wmax = max(y_pred) # line 100 multiclass_obj.cu
    wsum =0.0
    for i in y_pred : wsum +=  np.exp(i - wmax)             

    for r in range(y_pred.shape[0]):
        

         # TODO fix this, multiclass_obj.cu
        p = np.exp(y_pred[r]- wmax) / wsum
        
        target = y_true[r]

        # c = 1 if p >=0.5 else 0

        g = p - 1.0 if c == target else p
        # g = target - p
        # h = p * (1.0 -p)
        h = max((2.0 * p * (1.0 - p)).item(), 1e-6)
        # hess = np.vectorize(hessfunc)(y_pred)
        grad[r] = g
        hess[r] = h
    return grad, hess

def softprob_obj_tmp(y_true, y_pred):
    grad = np.zeros((y_pred.shape[0], y_pred.shape[1]), dtype=float) # for multi-class
    hess = np.zeros((y_pred.shape[0], y_pred.shape[1]), dtype=float) # for multi-class
    for rowid in range(y_pred.shape[0]):
        wmax = max(y_pred[rowid]) # line 100 multiclass_obj.cu
        wsum =0.0
        for i in y_pred[rowid] : wsum +=  np.exp(i - wmax)
        for c in range(y_pred.shape[1]):
            p = np.exp(y_pred[rowid][c]- wmax) / wsum
            target = y_true[rowid]
            g = p - 1.0 if c == target else p
            h = max((2.0 * p * (1.0 - p)).item(), 1e-6)
            grad[rowid][c] = g
            hess[rowid][c] = h
    return grad, hess