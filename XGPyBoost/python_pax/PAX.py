import numpy as np
from XGPyBoostClass import *
from customddsketch import DDSketch
from typing import Tuple
from treemodel import TreeNode
from params import Params
from tqdm import tqdm
from copy import deepcopy

from histograms import histogram
import time
from enum import Enum
import utils

from sketchtype import Sketch_type

class PAX:
    def __init__(self, params:Params) -> None:
        self.params:Params = params

    def fit(self, X:np.ndarray, y:np.ndarray, splits:list[list[float]]=None) -> None:
        """Fit the model in a federated fashion

        Args:
            self (PAX) : PAX instance
            X (np.ndarray): array of data np.ndarrays. X[0] being the dataset of node 0
            y (np.ndarray): array of target np.ndarrays. y[0] being the target dataset of node 0
            eA (float): Global Error Tolence Budget
            T (int): Maximum Number of Training Rounds
            l (callable): Model Loss Function
        """
        if splits is None:
            if self.params.sketch_type is Sketch_type.DDSKETCH:
                splits = utils.find_splits(X, self.params.eA, self.params.n_bins)
            elif self.params.sketch_type is Sketch_type.NORMAL: # TODO merge X
                splits = utils.data_to_histo(X)
            else:
                raise RuntimeError("implement other type here")

        n_classes = np.amax(y[0]) + 1

        amount_participants = len(X)
        P:list[PAXParticipant] = [PAXParticipant(i, X[i], y[i], self.params.n_trees) for i in range(amount_participants)]
        A:PAXAggregator = PAXAggregator(P, n_classes=n_classes, params=self.params, number_of_bins=self.params.n_bins)
        self.A = A

        # A.create_global_null_model(X[0].shape[0]) # line 1
        A.create_global_null_model_known_probas(X[0].shape[0], y)
        epsilonP:np.ndarray[float] = A.compute_local_epsilon(self.params.eA) # line 2 & 6

        for i in range(amount_participants): # line 4-8
            Pi:PAXParticipant = P[i]
            Pi.recieve_e(epsilonP[i]) # line 5
            # Pi.compute_histogram(number_of_bins=number_of_bins) # line 7
            Pi.splits = splits


        t = 0 # amount of trees counter

        # self.trees = [[] for i in range(n_classes)]

        for t in tqdm(range(1, self.params.n_trees), desc="> building trees"):
        # for t in range(1, T):
            DA = [] # line 11 # TODO initialize properly
            GA = [] # line 11 # TODO initialize properly
            HA = [] # line 11 # TODO initialize properly
            for i in range(amount_participants): # multithread?
                Pi:PAXParticipant = P[i]
                Pi.recieve_model(A.getmodel())
                Pi.predict()
                gpi, hpi = Pi.calculatedifferentials(self.params.objective)
                DXpi = Pi.getDXpi()
                DA.append(DXpi) # line 17 # TODO take union
                GA.append(gpi) # line 18 # TODO take union
                HA.append(hpi) # line 19 # TODO take union

            emA = min(epsilonP) # line 21

            DmA, GmA, HmA = A.merge_hist(DA, GA, HA, emA) # line 25
            # print("creating tree {}".format(t))
            A.grow_tree(DmA, GmA, HmA)


    def predict(self, X:np.ndarray):
        return self.A.predict(X)

    def predict_proba(self, X:np.ndarray):
        return self.A.predict_proba(X)

class PAXAggregator:

    def __init__(self, P:list, n_classes, params:Params, number_of_bins:int) -> None:
        self.P = P
        self.n_classes = n_classes
        self.params = params
        self.number_of_bins = number_of_bins
        self.trees = [] # features, numtrees

    def create_global_null_model(self, numfeatures:int) -> None:
        for i in range(self.n_classes):
            self.trees.append( [XGPyBoostClass.create_first_tree(numfeatures)] ) # features, numtrees

    def create_global_null_model_known_probas(self, numfeatures:int, y) -> None:
        """create first trees to have weights that use the probabilities of each respective class's probability.

        Args:
            y (_type_): y to retrieve the probailities of each class of
        """
        length = 0
        count = np.zeros(self.n_classes)
        for i in range(len(y)):
            length =+ y[i].shape[0]
            count =+ np.bincount(y[i])

        probas = count/length

        for i in range(self.n_classes):
            self.trees.append( [XGPyBoostClass.create_first_tree(numfeatures, probas[i])] )


    def compute_local_epsilon(self, eA:float)-> np.ndarray[float]:
        """Function compute_local_epsilon (line 29-40) from ong et al.

        Args:
            eA (float): Global Error Tolerance Budget

        Returns:
            np.ndarray[float]: array of Error Tolerance Budgets for every Participant E[0] belongs to P[0] for example
        """
        S = []
        for pi in self.P:
            S.append(pi.sendDataCount())
        E = []
        sumS = sum(S)
        for i in range(len(self.P)):
            ei = eA*(S[i]/ sumS)
            E.append(ei)
        return E

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], self.n_classes, self.params.n_trees))

        y_pred = np.zeros((X.shape[0], self.n_classes, self.params.n_trees))

        for c in range(self.n_classes):
            for i, tree in enumerate(self.trees[c]):
                y_pred[:, c, i] = tree.predict(X)



        # now y_preds are filled
        for i in range(self.params.n_trees):
            for rowid in range(y_pred.shape[0]):
                row = y_pred[rowid, : , i]
                wmax = max(row) # line 100 multiclass_obj.cu
                wsum =0.0
                for y in row : wsum +=  np.exp(y - wmax)
                probas[rowid,:, i] = np.exp(row -wmax) / wsum

        probas = np.average(probas, axis=2)
        return probas


    def predict(self, X):
        probas = self.predict_proba(X)

        highets_prob = np.zeros((X.shape[0], self.n_classes), dtype='int64')
        for i in range(X.shape[0]):
            highets_prob[i] = np.where(probas[i] == np.amax(probas[i]), 1, 0)

        return [ np.argmax(probdrow ) for probdrow in highets_prob ]

    def grow_tree(self, Dma, GmA, HmA):

        def fit_tree_thread(c):

            myDma = deepcopy(Dma)
            myGma = deepcopy(GmA[:, c])
            myHma = deepcopy(HmA[:, c])
            my_params = deepcopy(self.params)

            # start = time.time()
            tree = XGPyBoostClass._fit_tree(myDma, myGma, myHma, my_params) # THIS IS WHERE THE MULTITHREADING PROBLEM
            # end = time.time()
            # print(end-start)
            self.trees[c].append(tree)

        threads = []

        for c in range(self.n_classes):
            t:Thread = Thread(target=fit_tree_thread, args=(c,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        pass

    def merge_hist(self, DA, GA, HA, emA):
        # dit moet compleet anders. op plekken waar DA[pi] hetzelfde is moet de ga+ga en ha+ha!!!!
        DMA = np.zeros((self.number_of_bins, DA[0].shape[1]))
        DMA = np.ndarray([])
        GMA = [[] for i in range(DA[0].shape[1])]
        HMA = [[] for i in range(DA[0].shape[1])]
        n_participants = len(DA)

        #voor elke feature kijken welke splits unique splits er zijn en houd bij hoe vaak elke is genomen
        DMA = DA[0]
        for participant in range(1, len(DA)):
            DMA = np.concatenate((DMA, DA[participant]),axis=0)

        GMA = GA[0]
        for participant in range(1, len(GA)):
            GMA = np.concatenate((GMA, GA[participant]),axis=0)

        HMA = HA[0]
        for participant in range(1, len(HA)):
            HMA = np.concatenate((HMA, HA[participant]),axis=0)

        return DMA, GMA, HMA


    def getmodel(self):
        return [classestrees[-1] for classestrees in self.trees] # return the last model for every feature

class PAXParticipant:

    def __init__(self, id:int, X:np.ndarray, y, T:int) -> None:
        self.idk:int = id
        self.X:np.ndarray = X
        self.y:np.ndarray = y
        self.n_classes = np.amax(y) + 1
        self.DXpi = None # local histogram
        self.e:float = None
        self.model:list[TreeNode] = None  # feature, trees
        self.splits:list[list[float]]= None
        self.prediction:np.array[list[float]] = np.zeros((self.n_classes, X.shape[0])) # n_classes, X.shape[0]

    def getDXpi(self) -> object: # TODO change object to histogram
        return self.DXpi

    def compute_histogram(self, splits):
        pass

    def compute_histogram(self, number_of_bins:int, sketchtype:Sketch_type = Sketch_type.DDSKETCH) -> None: # TODO use self.X and self.e to construct local historgram on features. store it in self.DXpi

        match sketchtype:
            case Sketch_type.DDSKETCH:
                print("creating histogram for party {} 📊".format(self.idk))

                self.sketch = []
                self.histo = np.zeros((self.X.shape[1], number_of_bins))
                for feature in range(self.X.shape[1]):
                    self.sketch.append(DDSketch(self.e))
                    for x in self.X[:,feature]:
                        self.sketch[feature].add(x)

                    for i in range(1, number_of_bins+1):
                        self.histo[feature][i-1] = self.sketch[feature].get_quantile_value(i/number_of_bins)
            case Sketch_type.NORMAL:
                print("creating normal histogram with equal bins")





    def calculatedifferentials(self, l:callable) -> Tuple[np.ndarray, np.ndarray]: # use the prediction from self.prediction to calculate the differentials and store them in self.grad & self.hess
        # for n_class in range(self.n_classes):
        return l(self.y ,self.prediction.T)

    def recieve_model(self, model):
        self.model = model

    def predict(self): # TODO predict using the model and local histogram
        # use self.model to predict
        # use self.splits to find in which bin X would be and take the middle of that bin. this will be fed into the prediction model
        if self.DXpi is None: # this puts X into the bins!
            interp_values = np.zeros(self.X.shape)
            for feature_i in range(self.X.shape[1]):
                try:
                    bin_indices = np.digitize(self.X[:, feature_i], self.splits[feature_i])-1
                    for i, b in enumerate(bin_indices):
                        if b == 0:
                            interp_values[i, feature_i] = (self.splits[feature_i][0] + self.splits[feature_i][1])/2
                        else:
                            interp_values[i, feature_i] = (self.splits[feature_i][b-1] + self.splits[feature_i][b])/2 # TODO Double double check this method
                except ValueError: # all bins are the same, so just take the first one for all
                    interp_values[:, feature_i] = [self.splits[feature_i][0] for x in self.X[:, feature_i] ]
            self.DXpi = interp_values

        for class_i in range(self.n_classes):
            self.prediction[class_i] = self.model[class_i].predict(self.DXpi) # np.array(interp_values).T
        pass # DEBUG


    def sendDataCount(self):
        return self.X.shape[0]

    def recieve_e(self, ep):
        self.e = ep