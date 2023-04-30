import numpy as np
from params import Params
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
from threading import Thread
from treemodel import *



class XGPyBoostClass:
    def __init__(
        self,
        n_trees,
        obj,
        **kwargs
    ):
        self.obj = obj
        self.params = Params(n_trees, **kwargs)

    def create_first_tree(num_features):
        tree = TreeNode(np.full(num_features, True))
        tree.weight = 1
        tree.is_leaf = True
        return tree

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

        def fit_tree_thread(c):
            tree = self._fit_tree(X, grad[:, c], hess[:, c], self.params)
            self.trees[c].append(tree)
            preds[:, c] = tree.predict(X)


        for i in range(self.params.n_trees):
            threads = []
            grad, hess = self.obj(y, preds)
            print("starting with tree {}".format(i))

            for c in range(self.n_classes):
                t = Thread(target=fit_tree_thread, args=(c,))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()