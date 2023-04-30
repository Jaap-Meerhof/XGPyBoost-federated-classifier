from PAX import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from objectives import softprob

MAX_DEPTH = 6
N_TREES = 5
ETA = 1
GAMMA = 0.3 #std=0.3
MIN_CHILD_WEIGHT = 1 # std=1
REG_ALPHA=0 #std =0
REG_LAMBDA=1
N_PARTICIPANTS = 5
EA = 0.05
def main():

    print("starting tests")
    n_classes = 5
    X, y = make_classification(n_samples=int(10000) , n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)
    X_train = X_train[:10000]
    X_train = np.array_split(X_train, N_PARTICIPANTS)
    # X_test = np.array_split(X_test, N_PARTICIPANTS)
    y_train = np.array_split(y_train, N_PARTICIPANTS)
    # y_test = np.array_split(y_test, N_PARTICIPANTS)


    pax = PAX()
    pax.fit(X_train, y_train, EA, 100, softprob)


    pass



if __name__ == "__main__":


    main()