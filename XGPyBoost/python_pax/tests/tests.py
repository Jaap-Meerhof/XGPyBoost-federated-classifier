"""This file will test multiple setups in a row with the different parameters given.
"""
from test_helpers import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import Params

def main():
    test_fake_data()

def test_fake_data():
    print("starting tests")
    n_classes = 5
    X, y = make_classification(n_samples=int(10000) , n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    x_plot = [3, 4]
    params = [Params( n_trees=x, max_depth=6, eta=0.3, lam=1, alpha=0, gamma=0, min_child_weight=1, max_delta_step=0, n_participants=5) for x in x_plot]
    plot_accuracies(X_train, X_test, y_train, y_test, params, x_plot)

if __name__ == "__main__":
    main()
    # cProfile.run('main()', sort='cumtime')