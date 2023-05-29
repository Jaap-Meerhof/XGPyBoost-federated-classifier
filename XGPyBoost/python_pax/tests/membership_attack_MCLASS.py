from membership_helpers import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import Params
import utils
from PAX import PAX
from PAX import Sketch_type
N_CLASSES = 5
XSIZE = 10_000
SPLIT = 7_500 # XSIZE//2
N_TREES = 100

TARGET_MODEL_NAME = "target_modelMCLASS_500_100trees_eta1.pkl"
SAVE = True
def main():

    X, y = make_classification(n_samples=int(XSIZE) , n_features=20, n_informative=10, n_redundant=0, n_classes=N_CLASSES, random_state=50)

    shadow_fake = (X[:SPLIT, :], y[:SPLIT])
    X, y = X[SPLIT:, :], y[SPLIT:]
    splits = utils.data_to_histo(X)
    N_PARTICIPANTS = 5

    # target_model = MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
    # target_model.fit(X,y)


    X_PAX = np.array(np.array_split(X, N_PARTICIPANTS))
    y_PAX = np.array(np.array_split(y, N_PARTICIPANTS))
    target_model = None
    if SAVE and os.path.exists(TARGET_MODEL_NAME):
        print("> getting target model from pickle jar")
        target_model = pickle.load(open(TARGET_MODEL_NAME, "rb"))
    else:
        print("> creating target model as no pickle jar exists")
        target_model = PAX(Params(n_trees=N_TREES, max_depth=12, min_child_weight=1, lam=1, alpha=0, eta=1))
        target_model.fit(X_PAX, y_PAX, splits)
        pickle.dump(target_model, open( TARGET_MODEL_NAME, "wb"))

    # shadow_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=1000)
    # xgb.config_context(verbosity=3)


    # shadow_model = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=100)
    # attack_model = xgb.XGBClassifier(tree_method="exact", objective='binary:logistic', max_depth=6, n_estimators=10, learning_rate=0.3)
    attack_model = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=10)

    # attack_model = MLPClassifier(hidden_layer_sizes=(10,10), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
    y_pred = target_model.predict(X)
    tmp = target_model.predict_proba(X)
    print("> approx base accuracy: %.2f" % (accuracy_score(y, y_pred)))
    membership_inference_attack(shadow_fake=shadow_fake, target_model=target_model, shadow_model=shadow_model, attack_model=attack_model, X=X, n_classes=N_CLASSES)


if __name__ == "__main__":
    main()
    # cProfile.run('main()', sort='cumtime')