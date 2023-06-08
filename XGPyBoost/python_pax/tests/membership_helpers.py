import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from membershipresults import MembershipResults
import csv
import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PAX
from params import Params

def plot_histo(X):
    import matplotlib.pyplot as plt
    plt.hist(X, color='lightgreen', ec='black', bins=15)
    plt.show()
    pass

def plot_data(data: np.array, labels, destination= 'plot.png', name='Sample Text'):
    """will plot the different variables against the first column of data (the value tested against)

    Args:
        data (_type_): twodimentional array of variables
        labels (_type_): corresponding labels
    """
    import matplotlib.pyplot as plt
    import matplotlib.axis as Axis
    amount_of_plots = len(data[0])
    nrows = 3
    ncols = (amount_of_plots//3) + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(10,10))
    plt.subplots_adjust(hspace=0.4)

    x = data[:, 0]
    y = data[:, 1:]
    y_all = [list(tmp) for tmp in zip(*y)]

    for i, y in enumerate(y_all):
        ax= axs[i//ncols, i%ncols]
        ax.plot(x, y)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[i+1])
        ax.set_title(labels[i+1])

    ax = axs[(i+1)//ncols, (i+1)%ncols]
    ax.text(0.5, 0.5, name, 
        horizontalalignment='center', verticalalignment='center',
        fontsize=10, color='black')
    ax.set_xticks([])
    ax.set_yticks([])


    if nrows*ncols > amount_of_plots:
        for i in range(amount_of_plots, nrows, ncols):
            fig.delaxes[i//ncols, i%ncols]
    # plt.tight_layout()
    fig.suptitle(labels[0])
    # fig.text(ncols, nrows, 'testtesttest')
    plt.savefig(destination)
    # plt.title("test2")
    # plt.show()
    pass
# data = pickle.load(open("fulldata.pkl", "rb"))
# # data = np.array([[-1,1,2,3,4,5], [-2, 2,3,4,5,6]])
# # labels = ["N", "param1", "param2", "param3", "param4", "param5"]
# labels = ["acc_training_target", "acc_test_target", "overfit_target", 
#                 "acc_training_shadow", "acc_test_shadow", "overfit_shadow", 
#                 "acc_X_attack", "acc_other_attack", 
#                 "precision_50_attack", "acc_50_attack"]
# labels = ["N_TREES"] + labels
# # params = Params(10, 10, 0.1, 1, 1, 1, 1, eA = 0.2, n_bins=10, n_participants=7, num_class=5)
# MAX_DEPTH = 12
# N_TREES = 50
# ETA = 0.3
# GAMMA = 0.3 #std=0.3
# MIN_CHILD_WEIGHT = 1 # std=1
# REG_ALPHA=0 #std =0
# REG_LAMBDA= 1 #std =1
# N_PARTICIPANTS = 5
# N_BINS = 3
# EA = 1/N_BINS
# params = Params(N_TREES, MAX_DEPTH, ETA, REG_LAMBDA, REG_ALPHA, GAMMA, MIN_CHILD_WEIGHT, eA = EA, n_bins=N_BINS, n_participants=N_PARTICIPANTS, num_class=10)

# plot_data(np.array(data), labels, name=params.prettytext())

def split_shadowfake(shadow_fake):
    split = len(shadow_fake[0][:,0])//3 # 
    other_fake  = (shadow_fake[0][:split, :], shadow_fake[1][:split]) # splits the dataset
    test_fake   = (shadow_fake[0][split:2*split, :], shadow_fake[1][split:2*split])
    shadow_fake = (shadow_fake[0][2*split:, :], shadow_fake[1][2*split:]) # splits the datset
    return other_fake, test_fake, shadow_fake

def membership_inference_attack(shadow_fake, target_model:object, shadow_model, attack_model, X, orininal_y):
    # will do step B, C, and D from my paper

    pass # TODO step B, train shadow model on shadow_fake
    other_fake, test_fake, shadow_fake = split_shadowfake(shadow_fake)
    
    shadow_model.fit(shadow_fake[0], shadow_fake[1])
    y_pred = shadow_model.predict(test_fake[0])
    print("> shadow accuracy: %.2f" % (accuracy_score(test_fake[1], y_pred)))

    pass # TODO step C, train attack model on outputs of shadow_model and target_model on shadow_fake

    y0 = np.zeros(len(other_fake[0]))
    y1 = np.ones(len(shadow_fake[0]))

    x0 = other_fake[0]
    x1 = shadow_fake[0]
    x = np.concatenate((x0,x1), axis=0)
    y = np.concatenate((y0,y1), axis=0)
    indices = np.random.permutation(x.shape[0])
    x = x[indices]
    y = y[indices]


    attack_x_0 = shadow_model.predict_proba(np.array(x, dtype=float))
    # attack_x_1 = target_model.predict_proba(np.array(x))
    # tmp = np.max(attack_x_0, axis=1).reshape(-1, 1)
    tmp = attack_x_0
    # attack_x = np.column_stack((attack_x_0, attack_x_1))
    attack_model.fit(tmp,y)

    pass # TODO step D, check accuracy x when feeding it with real and fake data!
    # test_x = np.vstack((test_fake[0], X[0]))
    test_x = np.vstack((X, test_fake[0]))
    y = np.hstack( (np.ones(X.shape[0]), np.zeros(test_fake[0].shape[0])) )
    # predicted = np.column_stack((shadow_model.predict_proba(test_x), target_model.predict_proba(test_x)))
    # predicted = np.max(target_model.predict_proba(test_x), axis=1).reshape(-1,1)
    predicted = target_model.predict_proba(test_x)
    print(tmp.shape)
    print(predicted.shape)
    # y = np.hstack((np.zeros(test_fake[0].shape[0]), np.ones(X[0].shape[0]) ))
    y_pred = attack_model.predict(predicted)
    print("> Attack accuracy: %.2f" % (accuracy_score(y, y_pred)))
    print("> Attack precision: %.2f" % (precision_score(y, y_pred)))

    # y_pred = [ 1 if maxv > 0.9 else 0 for maxv in predicted ]
    # print("> Attack accuracy: %.2f" % (accuracy_score(y, y_pred)))
    # print("> Attack precision: %.2f" % (precision_score(y, y_pred)))


    # things to collect: 
    #   * accuracy target_model on training data

    data = []
    acc_training_target = accuracy_score(orininal_y, target_model.predict(X))
    data.append(acc_training_target)
    #   * accuracy target_model on test data
    acc_test_target = accuracy_score(shadow_fake[1], target_model.predict(shadow_fake[0])) # shadow_fake used for testing
    data.append(acc_test_target)
    #   * degree of overfitting
    overfit_target = acc_training_target - acc_test_target
    data.append(overfit_target)
    #   * accuracy shadow_model on training data
    acc_training_shadow = accuracy_score(shadow_fake[1], shadow_model.predict(shadow_fake[0]))
    data.append(acc_training_shadow)
    #   * accuracy shadow_model on test data
    acc_test_shadow = accuracy_score(test_fake[1], shadow_model.predict(test_fake[0]))
    data.append(acc_test_shadow)
    #   * degree of overfitting
    overfit_shadow = acc_training_shadow - acc_test_shadow
    data.append(overfit_shadow)
    #   * attack accuracy on X 

    acc_X_attack = accuracy_score(np.ones((X.shape[0],)), attack_model.predict(target_model.predict_proba(X)))
    data.append(acc_X_attack)
    #   * attack accuracy on other shadow
    acc_other_attack = accuracy_score(np.zeros((other_fake[1].shape[0],)), attack_model.predict(target_model.predict_proba(other_fake[0])))
    data.append(acc_other_attack)

    #   * precision of attack on both 50/50
    min = np.min((X.shape[0], other_fake[0].shape[0])) # takes the length of the shortest one
    fiftyfiftx = np.vstack((X[:min, :], other_fake[0][:min, :])) # TODO aggregate X and other_fake 50/50
    fiftyfifty = np.hstack((np.ones(min), np.zeros(min)))

    precision_50_attack = precision_score(fiftyfifty, attack_model.predict(target_model.predict_proba(fiftyfiftx)) )
    data.append(precision_50_attack)
    #   * accuracy both 50/50
    acc_50_attack = accuracy_score(fiftyfifty, attack_model.predict(target_model.predict_proba(fiftyfiftx)))
    data.append(acc_50_attack)
    # csv or json or plk pickle!
    return data

def getDNA():
    import pandas as pd
    df=pd.read_csv("/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/dna.csv")
    labels = df.columns
    shape = df.shape
    y = df["class"]
    X = df.drop(['class'], axis=1)

    X = np.array(X)
    y = np.array(y)
    return X, y

def getCensus(DATASETLOCATION):
    labels = pickle.load(open(DATASETLOCATION + "/census/census_feature_desc.p", "rb"))
    X = pickle.load(open("/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/census/census_features.p", "rb"))
    y = pickle.load(open("/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/census/census_labels.p", "rb"))
    return X, y, labels

def getCensusCloud():
    """downloads the Census dataset, this is a binary problem :(

    Returns:
        _type_: _description_
    """
    import urllib.request
    print("> downloading from JaapCloud1.0...")

    datalabel = urllib.request.urlopen("https://jaapmeerhof.nl/index.php/s/iNJPFMs34S99r3W/download").read() # label
    labels = pickle.loads(datalabel)

    dataX = urllib.request.urlopen("https://jaapmeerhof.nl/index.php/s/TrE8dBCcTs8dJWN/download").read() # X
    X = pickle.loads(dataX)

    datay = urllib.request.urlopen("https://jaapmeerhof.nl/index.php/s/Zpj4moWNy5tk7B5/download").read() # y
    y = pickle.loads(datay)
    print("> done downloading from JaapCloud1.0!")
    return X, y, labels
