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

N_CLASSES = 10
XSIZE = 40_000
SPLIT = 30_000 # XSIZE//2

TARGET_MODEL_NAME = "target_modelPURCHASE_2class_5n_200t_12d_400b.pkl"
TARGET_MODEL_NAME = "target_modelDebug.pkl"

SAVE = False
DATA_PATH = "/home/jaap/Documents/tmp/acquire-valued-shoppers-challenge/"
# DATA_PATH = "/home/hacker/cloud_jaap_meerhof/SchoolCloud/Master Thesis/Database/acquire-valued-shoppers-challenge/"
# DATA_PATH = '/data/BioGrid/meerhofj/acquire-valued-shoppers-challenge/'
DATA_PATH = '/data/BioGrid/meerhofj/acquire-valued-shoppers-challenge/'

MAX_DEPTH = 12
N_TREES = 50
ETA = 0.3
GAMMA = 0.3 #std=0.3
MIN_CHILD_WEIGHT = 1 # std=1
REG_ALPHA=0 #std =0
REG_LAMBDA= 1 #std =1
N_PARTICIPANTS = 5
N_BINS = 400
EA = 1/N_BINS


def main():
    full_data = []
    for N_TREES in [5, 10, 20, 30, 40, 50, 100]:
        random.seed(1)
        X = pickle.load(open(DATA_PATH+"purchase_100_features.p", "rb"))
        y = pickle.load(open(DATA_PATH+"purchase_100_10_labels.p", "rb"))
        random_indices = random.sample(range(X.shape[0]), XSIZE)
        X = X[random_indices]
        y = y[random_indices]
        # plot_histo(X[:, 0])
        # plot_histo(X[:, 1])

        params = Params(N_TREES, MAX_DEPTH, ETA, REG_LAMBDA, REG_ALPHA, GAMMA, MIN_CHILD_WEIGHT, eA = EA, n_bins=N_BINS, n_participants=N_PARTICIPANTS, num_class=N_CLASSES)
        
        
        shadow_fake = (X[:SPLIT, :], y[:SPLIT])
        X, y = X[SPLIT:, :], y[SPLIT:]
        splits = utils.data_to_histo(X, num_bins=N_BINS)

        # target_model = MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
        # target_model.fit(X,y)
        
        # target_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)

        # target_model.fit(X,y)


        X_PAX = np.array(np.array_split(X, N_PARTICIPANTS))
        y_PAX = np.array(np.array_split(y, N_PARTICIPANTS))

        target_model = None
        if SAVE and os.path.exists(TARGET_MODEL_NAME):
            print("> getting target model from pickle jar")
            target_model = pickle.load(open(TARGET_MODEL_NAME, "rb"))
        else:
            print("> creating target model as no pickle jar exists")
            target_model = PAX(Params(n_trees=N_TREES, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT, lam=REG_LAMBDA, alpha=REG_ALPHA, eta=ETA, gamma=GAMMA))
            # target_model = xgb.XGBClassifier(max_depth=MAX_DEPTH, tree_method='approx', objective="multi:softmax",
            #                 learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA)
            # target_model.fit(X,y)
            target_model.fit(X_PAX, y_PAX, splits)
            pickle.dump(target_model, open( TARGET_MODEL_NAME, "wb"))

        # shadow_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=1000)
        # xgb.config_context(verbosity=3)

        shadow_model = xgb.XGBClassifier(max_depth=MAX_DEPTH, tree_method='approx', objective="multi:softmax",
                            learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA)
        # shadow_model = xgb.XGBClassifier(tree_method="exact", objective='binary:logistic', num_class=N_CLASSES, max_depth=MAX_DEPTH, n_estimators=N_TREES, learning_rate=0.1)

        # shadow_model = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=100)
        attack_model = xgb.XGBClassifier(tree_method="exact", objective='binary:logistic', max_depth=6, n_estimators=50, learning_rate=0.3)
        # attack_model = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=10)

        # attack_model = MLPClassifier(hidden_layer_sizes=(10,10), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
        y_pred = target_model.predict(X)
        tmp = target_model.predict_proba(X)
        print("> base accuracy: %.2f" % (accuracy_score(y, y_pred)))
        data = membership_inference_attack(shadow_fake=shadow_fake, target_model=target_model, shadow_model=shadow_model, attack_model=attack_model, X=X, orininal_y=y)
        data = [N_TREES] + data
        labels = ["acc_training_target", "acc_test_target", "overfit_target", 
                "acc_training_shadow", "acc_test_shadow", "overfit_shadow", 
                "acc_X_attack", "acc_other_attack", 
                "precision_50_attack", "acc_50_attack"]
        labels = ["N_TREES"] + labels
        full_data.append(data)

        for i in range(len(labels)):
            print(labels[i] + " = " + str(data[i]))
        params.prettyprint()
    print(labels)
    print(full_data)
    pickle.dump(full_data, open( "fulldata_PURCHASE_N_BINS.pkl", "wb"))

    plot_data(np.array(full_data), labels, "purchase100"+ str(labels[0]) + ".png", params.prettytext())
    # data = [param] + data
    






if __name__ == "__main__":
    main()
    # cProfile.run('main()', sort='cumtime')