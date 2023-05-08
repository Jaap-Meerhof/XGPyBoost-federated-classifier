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


MAX_DEPTH = 6
N_TREES = 20
ETA = 1
GAMMA = 0.3 #std=0.3
MIN_CHILD_WEIGHT = 1 # std=1
REG_ALPHA=0 #std =0
REG_LAMBDA=1
N_PARTICIPANTS = 5
EA = 0.05
N_BINS = 255
def main():
    test_MNIST()
    # test_airline()
    test_make_classification()

def test_MNIST():
    print("testing MNIST")
    digits = datasets.load_digits()
    images = digits.images
    print(images[0])
    targets = digits.target
    print(targets[0])
    images = images.reshape(1797, 8*8)
    X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2)
    model = XGBClassifier(max_depth=MAX_DEPTH, tree_method='exact', objective="multi:softmax",
                           learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy xgb: %.2f%%" % (accuracy))

    sketch:list[DDSketch] = []
    splits:list[list[float]] = []
    #TODO make splits not using quantiles, but by using 255 same sized histograms.
    for feature in range(X_train.shape[1]):
        sketch.append(DDSketch(EA))
        df = pd.DataFrame(X_train[:, feature])
        for x in X_train[:,feature]:
            sketch[feature].add(x)
        splits.append([sketch[feature].get_quantile_value(i/N_BINS) for i in range(N_BINS)])
        pass
    # splits = data_to_histo(X_train)

    X_train = np.array_split(X_train, N_PARTICIPANTS)
    y_train = np.array_split(y_train, N_PARTICIPANTS)

    model = XGPyBoostClass(n_trees=N_TREES+20, obj=softprob, eta=ETA, gamma=GAMMA, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT)

    pax = PAX(model)
    pax.fit(X_train, y_train, EA, N_TREES+20, softprob, N_BINS, splits)

    preds_X = pax.predict(X_test)
    print('my python acc: ', accuracy_score(y_test, preds_X))

def test_airline():
    print("testing airline dataset")
    df1 = pd.read_csv("../../datasets/kaggle/airline/DelayedFlights.csv", delimiter=',', nrows=1000)
    df1.dataframeName = 'DelayedFlights.csv'
    nRow, nCol = df1.shape




def test_make_classification():
    print("starting tests")
    n_classes = 5
    X, y = make_classification(n_samples=int(10000) , n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)
    X_train = X_train[:10000]

    sketch:list[DDSketch] = []
    splits:list[list[float]] = []
    for feature in range(X_train.shape[1]):
        sketch.append(DDSketch(EA))
        for x in X_train[:,feature]:
            sketch[feature].add(x)
        splits.append([sketch[feature].get_quantile_value(i/N_BINS) for i in range(N_BINS)])



    ### NORMAL xgboost
    reg = xgb.XGBClassifier(max_depth=MAX_DEPTH, tree_method='exact', objective="multi:softmax", learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA) #tree_method="gpu_hist"
    reg.fit(X_train,y_train)
    preds_xgb = reg.predict(X_test)
    print('tree ', accuracy_score(y_test, preds_xgb))
    ### END NORMAL xgboost

    X_train = np.array_split(X_train, N_PARTICIPANTS)
    y_train = np.array_split(y_train, N_PARTICIPANTS)
    model = XGPyBoostClass(n_trees=N_TREES, obj=softprob, eta=ETA, gamma=GAMMA, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT)

    pax = PAX(model)
    pax.fit(X_train, y_train, EA, N_TREES, softprob, N_BINS, splits)

    preds_X = pax.predict(X_test)
    print('tree ', accuracy_score(y_test, preds_X))


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


    main()