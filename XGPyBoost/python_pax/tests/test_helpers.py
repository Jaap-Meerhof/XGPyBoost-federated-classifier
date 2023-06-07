import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from params import Params
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from PAX import PAX
import matplotlib.pyplot as plt

def run_both(X_train, X_test, y_train, y_test, params:Params):
    print("> running normal xgboost first....")
    model = XGBClassifier(max_depth=params.max_depth, tree_method='exact', objective="multi:softmax",
                           learning_rate=params.eta, n_estimators=params.n_trees, gamma=params.gamma, reg_alpha=params.alpha, reg_lambda=params.lam)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred)
    print("> Accuracy normal XGBoost: %.2f" % (accuracy_xgb))


    # splits:list[list[float]] = utils.find_splits(X_train, EA, N_BINS=N_BINS)
    splits:list[list[float]] = data_to_histo(X_train)

    n_participants = params.n_participants
    X_train = np.array_split(X_train, n_participants) # split evenly among participants
    y_train = np.array_split(y_train, n_participants) 

    print("> running federated XGBoost...")
    pax = PAX(params)
    pax.fit(X_train, y_train, splits)
    preds_X = pax.predict(X_test)
    accuracy_pax = accuracy_score(y_test, preds_X)
    print("> Accuracy federated XGBoost: %.2f" % (accuracy_pax))
    return accuracy_xgb, accuracy_pax

def data_to_histo(X):
    splits: list[list[float]] = []
    for feature in range(X.shape[1]):
        range_min = np.min(X)
        range_max = np.max(X)
        num_bins = 255
        bin_edges = np.linspace(range_min, range_max, num=num_bins-1)
        bin_indices = np.digitize(X[:, feature], bin_edges) -1
        bin_counts = np.bincount(bin_indices, minlength=num_bins)
        # plt.bar(range(num_bins), bin_counts, align='center')
        # plt.xticks(range(num_bins))
        # plt.xlabel('Bins')
        # plt.ylabel('Frequency')
        # plt.title('Histogram')
        # plt.show()
        splits.append(bin_edges)
    return splits

def plot_accuracies(X_train, X_test, y_train, y_test, params, plot_x):
    accuracies_xgb = []
    accuracies_pax = []
    for param in params:
        acc_xgb, acc_pax = run_both(X_train, X_test, y_train, y_test, param)
        accuracies_xgb.append(acc_xgb)
        accuracies_pax.append(acc_pax)
    
    plt.scatter(plot_x, accuracies_pax, s=10, c='r', label="pax")
    plt.scatter(plot_x, accuracies_xgb, s=10, c='g', label="xgboost")
    plt.title("accuracy pax vs xgb")
    plt.xlabel('n_trees')
    plt.ylabel('accuracy')
    plt.legend(loc="upper left")
    plt.show()
    return accuracies_xgb, accuracies_pax

