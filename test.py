import pytest
import numpy as np
import xgboost as xgb
from XGPyBoost.XGPyBoost import *

np.random.seed(1234)

# X = np.vstack((np.ones(5), np.ones(5)*2)).T
# y = np.ones(5)

class TestXGB:
    def test_default_params(self):
        #X = np.random.standard_normal(size = (10, 3))
        #y = X[:, 0]*2 + X[:, 1]*4 + X[:, 2]*3
        X = np.arange(1,7).reshape(-1,1)
        y = np.arange(1,7)
        dtrain = xgb.DMatrix(X, label = y)

        n_trees = 2
        model = XGPyBoost(n_trees = n_trees, obj = mse_obj, max_depth = 1, min_child_weight = 0, eta = 1, lam = 0) #, max_depth = 0, eta = 0.4, min_child_weight = 1)
        model.fit(X, y)
        xgbpy_preds = model.predict(X)
        print(xgbpy_preds)

        xgb_params = {
            'base_score' : 0,
            'objective' : 'reg:squarederror',
            'tree_method' : 'exact',
            'max_depth' : 1,
            'min_child_weight' : 0,
            'eta' : 1,
            'lambda' : 0
        }

        xgb_model = xgb.train(xgb_params, dtrain, n_trees)
        xgb_preds = xgb_model.predict(dtrain)
        print(xgb_preds)

        assert(np.allclose(xgbpy_preds, xgb_preds))

TestXGB().test_default_params()
