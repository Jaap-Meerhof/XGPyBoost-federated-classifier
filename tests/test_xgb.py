import pytest
import numpy as np
import xgboost as xgb
from XGPyBoost.XGPyBoost import *

np.random.seed(1234)

class TestXGB:
    def test_default_params(self):
        X = np.random.standard_normal(size = (1000, 4))
        y = X[:, 0]*2 + X[:, 1]*4 + X[:, 2]*3

        n_trees = 3
        model = XGPyBoost(n_trees = n_trees, obj = mse_obj)
        model.fit(X, y)
        xgbpy_preds = model.predict(X)

        xgb_params = {
            'base_score' : 0,
            'objective' : 'reg:squarederror',
            'tree_method' : 'exact',
            'n_estimators' : n_trees
        }

        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X, y)
        xgb_preds = xgb_model.predict(X)

        assert(np.allclose(xgbpy_preds, xgb_preds))

    def test_n_trees(self):
        X = np.random.standard_normal(size = (1000, 3))
        y = X[:, 0]*2 + X[:, 1]*4 + X[:, 2]*3

        n_trees = 10
        model = XGPyBoost(n_trees = n_trees, obj = mse_obj)
        model.fit(X, y)
        xgbpy_preds = model.predict(X)

        xgb_params = {
            'base_score' : 0,
            'objective' : 'reg:squarederror',
            'tree_method' : 'exact',
            'n_estimators' : n_trees
        }

        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X, y)
        xgb_preds = xgb_model.predict(X)

        assert(np.allclose(xgbpy_preds, xgb_preds))
        
    def test_max_depth(self):
        X = np.random.standard_normal(size = (1000, 3))
        y = X[:, 0]*2 + X[:, 1]*4 + X[:, 2]*3

        n_trees = 3
        model = XGPyBoost(n_trees = n_trees, obj = mse_obj, max_depth = 10)
        model.fit(X, y)
        xgbpy_preds = model.predict(X)

        xgb_params = {
            'base_score' : 0,
            'objective' : 'reg:squarederror',
            'tree_method' : 'exact',
            'n_estimators' : n_trees,
            'max_depth' : 10
        }

        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X, y)
        xgb_preds = xgb_model.predict(X)

        assert(np.allclose(xgbpy_preds, xgb_preds))


#TestXGB().test_default_params()
