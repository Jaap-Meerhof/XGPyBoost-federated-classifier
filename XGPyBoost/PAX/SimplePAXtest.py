from PAX import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from objectives import softprob
from customddsketch import DDSketch
from sklearn.metrics import accuracy_score

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

    X_train = np.array_split(X_train, N_PARTICIPANTS)
    # X_test = np.array_split(X_test, N_PARTICIPANTS)
    y_train = np.array_split(y_train, N_PARTICIPANTS)
    # y_test = np.array_split(y_test, N_PARTICIPANTS)

    model = XGPyBoostClass(n_trees=N_TREES, obj=softprob, eta=ETA, gamma=GAMMA, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT)

    pax = PAX(model)
    pax.fit(X_train, y_train, EA, N_TREES, softprob, N_BINS, splits)

    preds_X = pax.predict(X_test)
    print('tree ', accuracy_score(y_test, preds_X))

    pass



if __name__ == "__main__":


    main()