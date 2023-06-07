import pickle
import sys
import os
from membership_helpers import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import Params

labels = ["acc_training_target", "acc_test_target", "overfit_target", 
                "acc_training_shadow", "acc_test_shadow", "overfit_shadow", 
                "acc_X_attack", "acc_other_attack", 
                "precision_50_attack", "acc_50_attack"]
labels = ["N_TREES"] + labels

MAX_DEPTH = 12
N_TREES = 1
ETA = 0.3
GAMMA = 0.3 #std=0.3
MIN_CHILD_WEIGHT = 1 # std=1
REG_ALPHA=0 #std =0
REG_LAMBDA= 1 #std =1
N_PARTICIPANTS = 5

N_BINS = 400
EA = 1/N_BINS

full_data = pickle.load(open("fulldata_texas_n_trees.pkl", "rb"))
params = Params(N_TREES, MAX_DEPTH, ETA, REG_LAMBDA, REG_ALPHA, GAMMA, MIN_CHILD_WEIGHT, eA = EA, n_bins=N_BINS, n_participants=N_PARTICIPANTS, num_class=10)

plot_data(np.array(full_data), labels, "texas2_"+ str(labels[0]) + ".png", params.prettytext())