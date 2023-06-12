from membership_helpers import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from keras.datasets import mnist

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import Params
import utils
from PAX import PAX
from PAX import Sketch_type
N_CLASSES = 10
XSIZE = 1797
SPLIT = XSIZE//2
N_TREES = 10


MAX_DEPTH = 12
N_TREES = 50
ETA = 0.3
GAMMA = 0.3 #std=0.3
MIN_CHILD_WEIGHT = 1 # std=1
REG_ALPHA=0 #std =0
REG_LAMBDA= 1 #std =1
N_PARTICIPANTS = 5
N_BINS = 3
EA = 1/N_BINS
TARGET_MODEL_NAME = "target_modelMNIST_1_10trees_2_child_2lam_1alpha.pkl"
SAVE = False
N_PARTICIPANTS = 5

def main():
    full_data = []

    for N_TREES in [5, 10, 20, 30, 40, 50, 100]:
        params = Params(N_TREES, MAX_DEPTH, ETA, REG_LAMBDA, REG_ALPHA, GAMMA, MIN_CHILD_WEIGHT, eA = EA, n_bins=N_BINS, n_participants=N_PARTICIPANTS, num_class=10)

        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        digits = datasets.load_digits()
        # X = digits.images
        # X = X.reshape(1797, 8*8) # flatten
        # y = digits.target

        X = np.array(train_X).reshape(train_X.shape[0], 28*28)
        y = np.array(train_y)
        shadow_fake = (np.array(test_X).reshape(test_X.shape[0], 28*28), np.array(test_y))

        # shadow_fake = (X[:SPLIT, :], y[:SPLIT])
        # X, y = X[SPLIT:, :], y[SPLIT:]
        splits = utils.data_to_histo(X)
        
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
            # target_model = PAX(params)
            # target_model.fit(X_PAX, y_PAX, splits)
            target_model = xgb.XGBClassifier(max_depth=MAX_DEPTH, tree_method='approx', objective="multi:softmax",
                            learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA)
            target_model.fit(X,y)
            # target_model.fit(X_PAX, y_PAX, splits)
            pickle.dump(target_model, open( TARGET_MODEL_NAME, "wb"))

        # shadow_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=1000)
        # xgb.config_context(verbosity=3)

        shadow_model = xgb.XGBClassifier(max_depth=MAX_DEPTH, tree_method='approx', objective="multi:softmax",
                                learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA)
        
        # shadow_model = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=100)
        attack_model = xgb.XGBClassifier(tree_method="exact", objective='binary:logistic', max_depth=6, n_estimators=20, learning_rate=0.3)

        # attack_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
        n_classes = N_CLASSES
        y_pred = target_model.predict(X_PAX[0])
        print("> approx base accuracy: %.2f" % (accuracy_score(y_PAX[0], y_pred)))
        data = membership_inference_attack(shadow_fake=shadow_fake, target_model=target_model, shadow_model=shadow_model, attack_model=attack_model, X=X, orininal_y=y)
        data = [N_TREES] + data
        full_data.append(data)

    labels = ["acc_training_target", "acc_test_target", "overfit_target", 
                "acc_training_shadow", "acc_test_shadow", "overfit_shadow", 
                "acc_X_attack", "acc_other_attack", 
                "precision_50_attack", "acc_50_attack"]
    labels = ["N_TREES"] + labels
    print(labels)
    print(full_data)
    pickle.dump(full_data, open( "fulldata_MNIST_n_trees.pkl", "wb"))
    params = Params(N_TREES, MAX_DEPTH, ETA, REG_LAMBDA, REG_ALPHA, GAMMA, MIN_CHILD_WEIGHT, eA = EA, n_bins=N_BINS, n_participants=N_PARTICIPANTS, num_class=10)
    
    plot_data(np.array(full_data), labels, "MNIST_"+ str(labels[0]) + ".png", params.prettytext())

if __name__ == "__main__":
    main()
    # cProfile.run('main()', sort='cumtime')