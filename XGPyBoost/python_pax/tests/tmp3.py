import xgboost as xgb
import numpy as np

from membership_helpers import *

data = pickle.load(open("fulldata_MNIST_n_trees.pkl", "rb"))
# data = np.array([[-1,1,2,3,4,5], [-2, 2,3,4,5,6]])
# labels = ["N", "param1", "param2", "param3", "param4", "param5"]
labels = ["acc_training_target", "acc_test_target", "overfit_target", 
                "acc_training_shadow", "acc_test_shadow", "overfit_shadow", 
                "acc_X_attack", "acc_other_attack", 
                "precision_50_attack", "acc_50_attack"]
labels = ["N_TREES"] + labels
# params = Params(10, 10, 0.1, 1, 1, 1, 1, eA = 0.2, n_bins=10, n_participants=7, num_class=5)
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
params = Params(N_TREES, MAX_DEPTH, ETA, REG_LAMBDA, REG_ALPHA, GAMMA, MIN_CHILD_WEIGHT, eA = EA, n_bins=N_BINS, n_participants=N_PARTICIPANTS, num_class=10)

plot_data(np.array(data), labels, name=params.prettytext(), suptext="MNIST N_trees")