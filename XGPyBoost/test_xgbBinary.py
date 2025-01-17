import pytest
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from XGPyBoostBinary import *

np.random.seed(1234)

def main():
    print("starting tests")
    n_classes = 2
    X, y = make_classification(n_samples=2250, n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)
    model = XGPyBoostBinary(n_trees=5, obj=logistic_obj, eta=0.3, gamma=0.5, max_depth=6, min_child_weight=1.0, )
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    # print(preds)
    # print(y)
    print('tree: ', accuracy_score(y_test, preds))
    # print(preds)


    reg = xgb.XGBClassifier() #tree_method="gpu_hist"
    reg.fit(X_train,y_train)
    preds_xgb = reg.predict(X_test)
    print('treexgb ', accuracy_score(y_test, preds_xgb))
        

    

if __name__ == "__main__":
    main()