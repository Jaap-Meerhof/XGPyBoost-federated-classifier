import pytest
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from XGPyBoostMulti import *

np.random.seed(1234)

MAX_DEPTH = 6
N_TREES = 5
ETA = 1
GAMMA = 1 #std=0.3
MIN_CHILD_WEIGHT = 1 # std=1
REG_ALPHA=0 #std =0
REG_LAMBDA=1


def main():
    print("starting tests")
    n_classes = 5
    X, y = make_classification(n_samples=2250, n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)

    model = XGPyBoostMulti(n_trees=N_TREES, obj=softprob_obj_tmp, eta=ETA, gamma=GAMMA, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT)
    model.fit(X_train,y_train)
    print(y[:10])

    preds = model.predict(X_test)
    print(preds[:10])
    print('tree: ', accuracy_score(y_test, preds))


    reg = xgb.XGBClassifier(max_depth=MAX_DEPTH, tree_method='exact', objective="multi:softmax", learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA) #tree_method="gpu_hist"
    reg.fit(X_train,y_train)
    preds_xgb = reg.predict(X_test)
    print('tree ', accuracy_score(y_test, preds_xgb))
    

    

if __name__ == "__main__":
    main()