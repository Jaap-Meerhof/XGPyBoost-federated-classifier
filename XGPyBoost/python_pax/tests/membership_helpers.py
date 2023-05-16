import random
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PAX

def membership_inference_attack(shadow_fake, target_model:PAX, shadow_model, attack_model, X, n_classes):
    # will do step B, C, and D from my paper

    pass # TODO step B, train shadow model on shadow_fake
    # TODO split up shadow_fake into shadow_fake and other_fake!
    split = len(shadow_fake[0][:,0])/3 # split them halfway
    other_fake  = (shadow_fake[0][:split, :], shadow_fake[1][:split, :]) # splits the dataset
    test_fake   = (shadow_fake[0][split:2*split, :], shadow_fake[1][split:2*split, :])
    shadow_fake = (shadow_fake[0][2*split:, :], shadow_fake[1][2*split:, :]) # splits the datset
    
    shadow_model.fit(shadow_fake[0], shadow_fake[1])

    pass # TODO step C, train attack model on outputs of shadow_model and target_model on shadow_fake
    x = []
    y = []
    for i in range(len(other_fake[0]) + len(shadow_fake[0][:,0])):
        rand = random.choice(range(2)) # random value between 1 and 0
        if rand is 0:
            x.append.other_fake[0].pop(rand)
            y.append(0)
        else:
            x.append.shadow_fake[0].pop(rand)
            y.append(1)
    attack_x = np.zeros((len(x)[:,0], n_classes))

    attack_x[:,0] = shadow_fake.predict_proba(x)
    attack_x[:,1] = target_model.predict_proba(x)

    attack_model.fit(attack_x,y)

    pass # TODO step D, check accuracy x when feeding it with real and fake data!
    test_x = np.concatenate(test_fake[0], X)
    y = np.concatenate(np.zeros(len(test_fake[0]), np.ones(len(X[:,0]))))
    attack_model.predict()