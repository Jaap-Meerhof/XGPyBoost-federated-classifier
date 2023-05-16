from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import numpy as np

MAX_DEPTH = 6
N_TREES = 100
ETA = 1
GAMMA = 0.3 #std=0.3
MIN_CHILD_WEIGHT = 1 # std=1
REG_ALPHA=0 #std =0
REG_LAMBDA=1
N_PARTICIPANTS = 1

N_BINS = 255
EA = 1/N_BINS\


def test_make_classification():
    print("starting tests")
    n_classes = 5
    X, y = make_classification(n_samples=int(40_000) , n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=43)
    
    X_adv, y_adv = X[10_000:, :], y[10_000:]
    X_shadow, y_shadow = X[10_000:20_000, :], y[10_000:20_000]
    X_not, y_not = X[20_000:30_000, :], y[20_000:30_000]
    X, y = X[30_000:40_000, :], y[30_000:40_000]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)
    X_train = X_train[:10000]


    print("> running normal xgboost first....")
    target = XGBClassifier(max_depth=MAX_DEPTH, tree_method='exact', objective="multi:softmax",
                           learning_rate=ETA, n_estimators=N_TREES, gamma=GAMMA, reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA)
    target.fit(X_train, y_train)
    y_pred=target.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("> Accuracy normal XGBoost: %.2f" % (accuracy))

    # X_shadow, y_shadow = make_classification(n_samples=int(10000) , n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=42)

    X_train_s, X_test_s, y_train_s, y_test_s= train_test_split(X_shadow, y_shadow, test_size=0.2, random_state=420)

    # shadow = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=40).fit(X_train_s, y_train_s)
    shadow = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000).fit(X_train_s, y_train_s)

    #now we have to create a dataset containing, X=[shadow(input), target(inpX_shadowut)] with y= [in_shadow_train_yes_no]

    # X_adv, y_adv= make_classification(n_samples=1000, n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=420)
    inputs = np.vstack((X_train_s, X_adv))
    flags = np.hstack((np.ones(len(X_train_s)), np.zeros(len(X_adv))))

    predshadow = shadow.predict_proba(inputs)
    predclf = target.predict_proba(inputs)

    X_adv_output = [ (max(predshadow[i]), max(predclf[i])) for i in range(len(inputs))]
    print(np.shape(X_adv_output))
    print(np.shape(X_train))

    # attack_model = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=40).fit(X_adv_output, flags)
    attack_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000).fit(X_adv_output, flags)
    predshadow = shadow.predict_proba(X_train)
    predclf = target.predict_proba(X_train)

    # X_not, y_not= make_classification(n_samples=10000, n_features=20, n_informative=4, n_redundant=0, n_classes=n_classes, random_state=420)


    predshadow_not = shadow.predict_proba(X_not)
    predclf_not = target.predict_proba(X_not)

    X_adv_output = [ (max(predshadow[i]), max(predclf[i])) for i in range(len(X_train))]
    for i in range (len(predshadow_not)):
        X_adv_output.append((max(predshadow_not[i]), max(predclf_not[i])))
    
    y_true = np.concatenate((np.ones(len(X_train)), np.zeros(len(X_not))))

    y_pred = attack_model.predict(X_adv_output)
    accuracy = accuracy_score(y_pred, y_true)

    print(f"Accuracy: {accuracy}")


test_make_classification()
