import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PAX

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
    x = []
    y = []
    for i in range(len(other_fake[0]) + len(shadow_fake[0][:,0])):
        rand = random.choice(range(2)) # random value between 1 and 0
        if rand == 0:
            x.append(other_fake[0][rand,:])
            np.delete(other_fake[0], rand)
            y.append(0)
        else:
            x.append(shadow_fake[0][rand,:])
            np.delete(shadow_fake[0], rand)
            y.append(1)
    attack_x = np.zeros((len(x), n_classes))

    attack_x_0 = shadow_model.predict_proba(np.array(x, dtype=float))
    attack_x_1 = target_model.predict_proba(np.array(x))
    tmp = np.max(attack_x_0, axis=1)
    attack_x = np.column_stack((attack_x_0, attack_x_1))
    attack_model.fit(attack_x,y)

    pass # TODO step D, check accuracy x when feeding it with real and fake data!
    test_x = np.vstack((test_fake[0], X[0]))
    predicted = np.column_stack((shadow_model.predict_proba(test_x), target_model.predict_proba(test_x)))
    # predicted = target_model.predict_proba(test_x)
    y = np.hstack((np.zeros(test_fake[0].shape[0]), np.ones(X[0].shape[0]) ))
    y_pred = attack_model.predict(predicted)
    print("> Attack accuracy: %.2f" % (accuracy_score(y, y_pred)))
    print("> Attack precision: %.2f" % (precision_score(y, y_pred)))
