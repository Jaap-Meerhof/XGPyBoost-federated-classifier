from membership_helpers import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import Params

N_CLASSES = 5
XSIZE = 40_000
SPLIT = 40_00/2
def main():
    pass
    X, y = make_classification(n_samples=int(40_000) , n_features=20, n_informative=4, n_redundant=0, n_classes=N_CLASSES, random_state=43)
    
    shadow_fake = (X[:SPLIT, :], y[:SPLIT, :])
    X, y = X[SPLIT:, :], y[SPLIT:, :]

    target_model = PAX(Params(n_trees=10)).fit(X, y, splits)

    shadow_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000)
    attack_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000)
    n_classes = N_CLASSES
    membership_inference_attack(shadow_fake=shadow_fake, target_model=target_model, shadow_model=shadow_model, attack_model=attack_model, X=X, n_classes=n_classes)


if __name__ == "__main__":
    main()
    # cProfile.run('main()', sort='cumtime')