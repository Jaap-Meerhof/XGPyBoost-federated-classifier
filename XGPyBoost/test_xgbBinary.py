import pytest
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from XGPyBoostBinary import *

np.random.seed(1234)

def main():
    print("starting tests")
    n_classes = 2
    X, y = make_classification(n_samples=1250, n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=42)

    model = XGPyBoostBinary(n_trees=5, obj=softprob_obj, eta=0.3, gamma=0.5, max_depth=6, min_child_weight=1.0, )
    model.fit(X,y)
    preds = model.predict(X)
    print(preds)
    print(y)
    print('tree: ', accuracy_score(y, preds))
    print(preds)

    # reg = xgb.XGBClassifier() #tree_method="gpu_hist"

    # reg.fit(X,y)

    

if __name__ == "__main__":
    main()