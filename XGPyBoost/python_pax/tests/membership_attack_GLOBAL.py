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

# TO EDIT WITH TESTS
N_CLASSES = 100
XSIZE = 80_000  # 3_186
SPLIT = XSIZE-(XSIZE//4) # XSIZE//2

 # TEXAS TEXAS
# TESTNAME = "TEXAS_N_TREES_CENTRAL_12DEPTH_20k"
# PLOT_TITLE = "TEXAS, centralised NTREES"
 
 # PURCHASE PURCHASE
TESTNAME = "PURCHASE10_REG_MIN_CHILD_WEIGHT3.0_CENTRAL"
PLOT_TITLE = "PURCHASE 10, centralised MIN_CHILD_WEIGHT3.0"

 # MCLASS MCLASS
# TESTNAME = "MCLASS100_N_TREES_CENTRAL_12D"
# PLOT_TITLE = "MCLASS 100, centralised NTREES"

# UNIVERAL
TARGET_MODEL_NAME = "target_model_" + TESTNAME + ".pkl"
FULLDATA_NAME = "fulldata/fulldata_" + TESTNAME + ".pkl"
PLOT_NAME = "plot_" + TESTNAME + ".png"

# DATA_PATH = "/home/jaap/Documents/tmp/acquire-valued-shoppers-challenge/"
# DATA_PATH = "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/texas/"
# DATA_PATH = "/home/hacker/cloud_jaap_meerhof/SchoolCloud/Master Thesis/Database/acquire-valued-shoppers-challenge/"
DATA_PATH = '/data/BioGrid/meerhofj/texas/'
# getdata = 
SAVE = False
MAX_DEPTH = 12
N_TREES = 100
ETA = 0.3
GAMMA = 0.3 #std=0.3
MIN_CHILD_WEIGHT = 1 # std=1
REG_ALPHA=0 #std =0
REG_LAMBDA= 1 #std =1
N_PARTICIPANTS = 5

N_BINS = 400
EA = 1/N_BINS

# X = pickle.load(open(DATA_PATH+"texas_100_v2_features.p", "rb"))
# y = pickle.load(open(DATA_PATH+"texas_100_v2_labels.p", "rb"))
# X_, y_, _ = getTEXAS()
X_, y_, _ = getPURCHASE(10)
N_CLASSES = 10
XSIZE = 40_000
SPLIT = 30_000 

# X_, y_ = make_classification(n_samples=int(XSIZE) , n_features=20, n_informative=10, n_redundant=0, n_classes=N_CLASSES, random_state=50)

def main():
    full_data = []

    for MIN_CHILD_WEIGHT in [0, 0.1, 0.3, 1, 3, 8, 15, 50]:
        param = MIN_CHILD_WEIGHT
        paramstr = "MIN_CHILD_WEIGHT" + str(param)
        paramname = "MIN_CHILD_WEIGHT"
        random.seed(1)
        # X, y = getDNA()
        # X, y, _ = getCensusCloud()
        # X = pickle.load(open(DATA_PATH+"texas_100_v2_features.p", "rb"))
        # y = pickle.load(open(DATA_PATH+"texas_100_v2_labels.p", "rb"))
        random_indices = random.sample(range(X_.shape[0]), XSIZE)
        X, y= X_[random_indices], y_[random_indices]

        X = np.array(X)
        y = np.array(y)
        params = Params(N_TREES, MAX_DEPTH, ETA, REG_LAMBDA, REG_ALPHA, GAMMA, MIN_CHILD_WEIGHT, eA = EA, n_bins=N_BINS, n_participants=N_PARTICIPANTS, num_class=N_CLASSES)

        
        # plot_histo(X)
        # X = purchase100['features'][random_indices]
        # y = purchase100['labels'][random_indices]
        # X, y = make_classification(n_samples=int(XSIZE) , n_features=20, n_informative=10, n_redundant=0, n_classes=N_CLASSES, random_state=50)

        shadow_fake = (X[:SPLIT, :], y[:SPLIT])
        X, y = X[SPLIT:, :], y[SPLIT:]
        splits = utils.data_to_histo(X)

        # target_model = MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
        # target_model.fit(X,y)
        
        # target_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)

        # target_model.fit(X,y)


        # X_PAX = np.array(np.array_split(X, N_PARTICIPANTS))
        # y_PAX = np.array(np.array_split(y, N_PARTICIPANTS))
        def insert_paramname():
            return "models/" + TARGET_MODEL_NAME[:-4] + "_" + paramstr + TARGET_MODEL_NAME[-4:]
        
        target_model = None
        if SAVE and os.path.exists(insert_paramname()):
            print("> getting target model from pickle jar")
            target_model = pickle.load(open(insert_paramname(), "rb"))
        else:
            print("> creating target model as no pickle jar exists")
            # target_model = PAX(Params(n_trees=N_TREES, max_depth=MAX_DEPTH, min_child_weight=MIN_CHILD_WEIGHT, lam=REG_LAMBDA, alpha=REG_ALPHA, eta=ETA, gamma=GAMMA))
            target_model = xgb.XGBClassifier(max_depth=MAX_DEPTH, tree_method='approx', objective="multi:softmax",
                            learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA, min_child_weight = MIN_CHILD_WEIGHT)
            target_model.fit(X,y)
            # target_model.fit(X_PAX, y_PAX, splits)
            pickle.dump(target_model, open( insert_paramname(), "wb"))

        # shadow_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=1000)
        # xgb.config_context(verbosity=3)

        shadow_model = xgb.XGBClassifier(max_depth=MAX_DEPTH, tree_method='approx', objective="multi:softmax", # "multi:softmax"
                            learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA, min_child_weight = MIN_CHILD_WEIGHT)
        # shadow_model = xgb.XGBClassifier(tree_method="exact", objective='binary:logistic', num_class=N_CLASSES, max_depth=MAX_DEPTH, n_estimators=N_TREES, learning_rate=0.1)

        # shadow_model = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=100)
        attack_model = xgb.XGBClassifier(tree_method="exact", objective='binary:logistic', max_depth=8, n_estimators=50, learning_rate=0.3)
        # attack_model = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=10)

        # attack_model = MLPClassifier(hidden_layer_sizes=(10,10), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
        y_pred = target_model.predict(X)
        tmp = target_model.predict_proba(X)
        print("> approx base accuracy: %.2f" % (accuracy_score(y, y_pred)))
        data = membership_inference_attack(shadow_fake=shadow_fake, target_model=target_model, shadow_model=shadow_model, attack_model=attack_model, X=X, orininal_y=y)
        data = [param] + data

        
        full_data.append(data)

        # for i in range(len(labels)):
        #     print(labels[i] + " = " + str(data[i]))
        params.prettyprint()
    labels = ["acc_training_target", "acc_test_target", "overfit_target", 
                "acc_training_shadow", "acc_test_shadow", "overfit_shadow", 
                "acc_X_attack", "acc_other_attack", 
                "precision_50_attack", "acc_50_attack"]
    labels = [paramname] + labels
    print(labels)
    print(full_data)
    pickle.dump(full_data, open( FULLDATA_NAME, "wb"))
    params = Params(N_TREES, MAX_DEPTH, ETA, REG_LAMBDA, REG_ALPHA, GAMMA, MIN_CHILD_WEIGHT, eA = EA, n_bins=N_BINS, n_participants=N_PARTICIPANTS, num_class=10)
    
    plot_data(np.array(full_data), labels, PLOT_NAME, params.prettytext(), suptext= PLOT_TITLE)


if __name__ == "__main__":
    main()
    # cProfile.run('main()', sort='cumtime')main