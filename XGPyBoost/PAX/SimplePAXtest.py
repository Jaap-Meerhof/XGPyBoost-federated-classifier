from PAX import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from objectives import softprob
from customddsketch import DDSketch
from customddsketch import LogCollapsingLowestDenseDDSketch
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from keras.datasets import cifar10
import utils
import pickle
import os

import cProfile # DEBUG
import time  # DEBUG

MAX_DEPTH = 6
N_TREES = 10
ETA = 1
GAMMA = 0.3 #std=0.3
MIN_CHILD_WEIGHT = 1 # std=1
REG_ALPHA=0 #std =0
REG_LAMBDA=1
N_PARTICIPANTS = 1

N_BINS = 255
EA = 1/N_BINS
def main():
    # test_cifar10()
    test_MNIST()
    # test_airline()
    # test_make_classification()

def test_MNIST():
    print("testing MNIST")
    digits = datasets.load_digits()
    images = digits.images
    print(images[0])
    targets = digits.target
    print(targets[0])
    images = images.reshape(1797, 8*8)
    X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2)
    run_both(X_train, X_test, y_train, y_test)

def run_both(X_train, X_test, y_train, y_test):
    print("> running normal xgboost first....")
    model = XGBClassifier(max_depth=MAX_DEPTH, tree_method='exact', objective="multi:softmax",
                           learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("> Accuracy normal XGBoost: %.2f" % (accuracy))


    splits:list[list[float]] = utils.find_splits(X_train, EA, N_BINS=N_BINS)

    X_train = np.array_split(X_train, N_PARTICIPANTS)
    y_train = np.array_split(y_train, N_PARTICIPANTS)
    print("> running federated XGBoost...")
    model = XGPyBoostClass(n_trees=N_TREES, obj=softprob, eta=ETA, gamma=GAMMA, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT)
    pax = PAX(model)
    pax.fit(X_train, y_train, EA, N_TREES, softprob, N_BINS, splits)

    preds_X = pax.predict(X_test)
    print("> Accuracy normal XGBoost: %.2f" % (accuracy_score(y_test, preds_X)))

def test_iris():
    iris = datasets.load_iris()


def test_cifar10():
    print("testing cifar10")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # X_train is 50000 rows of 3x32x32 values --> reshaped in 50000 x 3072
    RESHAPED = 3072

    X_train = X_train.reshape(50000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # normalize the datasets
    X_train /= 255.
    X_test /= 255.

    # model = XGBClassifier(max_depth=MAX_DEPTH, tree_method='exact', objective="multi:softmax",
    #                        learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA)
    # model.fit(X_train, y_train)
    # y_pred=model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy xgb: %.2f%%" % (accuracy))
     # 94% acccuracy
    X_train = X_train[:10000]
    file_path = 'cifar10.kl'
    splits:list[list[float]] = None

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            splits = pickle.load(file)
    else:
        splits = utils.find_splits(X_train, EA, N_BINS=N_BINS)
        with open('splits.pkl', 'wb') as file:
            pickle.dump(splits)


    print("sketching done")
    X_train = np.array_split(X_train, N_PARTICIPANTS)
    y_train = np.array_split(y_train, N_PARTICIPANTS)

    model = XGPyBoostClass(n_trees=N_TREES, obj=softprob, eta=ETA, gamma=GAMMA, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT)

    pax = PAX(model)
    pax.fit(X_train, y_train, EA, N_TREES, softprob, N_BINS, splits)

    preds_X = pax.predict(X_test)
    print('my python acc: ', accuracy_score(y_test, preds_X))

def test_airline():
    print("testing airline dataset")
    df1 = pd.read_csv("datasets/kaggle/airline/DelayedFlights.csv", delimiter=',', nrows=100000)
    df1 = df1.drop("Unnamed: 0",1)
    df1.dataframeName = 'DelayedFlights.csv'
    nRow, nCol = df1.shape
    leaky_features = ["Year", "Diverted", "ArrTime", "ActualElapsedTime", "AirTime", "ActualElapsedTime", "AirTime", "ArrDelay", "TaxiIn", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay","LateAircraftDelay", "CancellationCode"]
    target = ["LateAircraftDelay"]
    features = [x for x in df1.columns if (x != target[0]) & (x not in leaky_features) & (len(df1[x].unique().tolist()) > 1)]


    def get_dtypes(data,features):
        output = {}
        for f in features:
            dtype = str(data[f].dtype)
            if dtype not in output.keys(): output[dtype] = [f]
            else: output[dtype] += [f]
        return output
    dtypes = get_dtypes(df1,features)
    categories = ["Month", "DayOfWeek", "DayofMonth"]
    categories += dtypes["object"]
    numerics = [i for i in dtypes["int64"] if i not in categories]
    numerics += dtypes["float64"]
    for numeric in numerics: df1[numeric] = df1[numeric].fillna(0)
    categories.remove("TailNum")
    # from pandas import get_dummies
    # from pandas import concat
    # one_hot_encoded = get_dummies(df1[categories].fillna("Unknown"))
    # X = concat([one_hot_encoded, df1[numerics].fillna(0)],1)
    X = df1[numerics, categories]
    pass


def test_make_classification():
    print("starting tests")
    n_classes = 5
    X, y = make_classification(n_samples=int(10000) , n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)
    X_train = X_train[:10000]

    run_both(X_train, X_test, y_train, y_test)


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

if __name__ == "__main__":
    # main()
    cProfile.run('main()', sort='cumtime')