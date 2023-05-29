import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PAX

def plot_histo(X):
    import matplotlib.pyplot as plt
    plt.hist(X[0], color='lightgreen', ec='black', bins=15)
    plt.show()
    pass

def membership_inference_attack(shadow_fake, target_model:PAX, shadow_model, attack_model, X, n_classes):
    # will do step B, C, and D from my paper

    pass # TODO step B, train shadow model on shadow_fake
    # TODO split up shadow_fake into shadow_fake and other_fake!
    split = len(shadow_fake[0][:,0])//3 # split them halfway
    other_fake  = (shadow_fake[0][:split, :], shadow_fake[1][:split]) # splits the dataset
    test_fake   = (shadow_fake[0][split:2*split, :], shadow_fake[1][split:2*split])
    shadow_fake = (shadow_fake[0][2*split:, :], shadow_fake[1][2*split:]) # splits the datset

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
    tmp = np.max(attack_x_0, axis=1).reshape(-1, 1)
    # tmp = attack_x_0
    # attack_x = np.column_stack((attack_x_0, attack_x_1))
    attack_model.fit(tmp,y)

    pass # TODO step D, check accuracy x when feeding it with real and fake data!
    # test_x = np.vstack((test_fake[0], X[0]))
    test_x = np.vstack((X, test_fake[0]))
    y = np.hstack( (np.ones(X.shape[0]), np.zeros(test_fake[0].shape[0])) )
    # predicted = np.column_stack((shadow_model.predict_proba(test_x), target_model.predict_proba(test_x)))
    predicted = np.max(target_model.predict_proba(test_x), axis=1).reshape(-1,1)
    # predicted = target_model.predict_proba(test_x)
    # y = np.hstack((np.zeros(test_fake[0].shape[0]), np.ones(X[0].shape[0]) ))
    y_pred = attack_model.predict(predicted)
    print("> Attack accuracy: %.2f" % (accuracy_score(y, y_pred)))
    print("> Attack precision: %.2f" % (precision_score(y, y_pred)))

    y_pred = [ 1 if maxv > 0.9 else 0 for maxv in predicted ]
    print("> Attack accuracy: %.2f" % (accuracy_score(y, y_pred)))
    print("> Attack precision: %.2f" % (precision_score(y, y_pred)))
    pass

