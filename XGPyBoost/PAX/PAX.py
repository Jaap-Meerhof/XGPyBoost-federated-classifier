import numpy as np
from XGPyBoostClass import *
from customddsketch import DDSketch
from typing import Tuple

class PAX:
    def __init__(self) -> None:
        pass

    def fit(self, X:np.ndarray, y:np.ndarray, eA:float, T:int, l:callable) -> None:
        """Fit the model in a federated fashion

        Args:
            self (PAX) : PAX instance
            X (np.ndarray): array of data np.ndarrays. X[0] being the dataset of node 0
            y (np.ndarray): array of target np.ndarrays. y[0] being the target dataset of node 0
            eA (float): Global Error Tolence Budget
            T (int): Maximum Number of Training Rounds
            l (callable): Model Loss Function
        """
        amount_participants = len(X)
        P:list[PAXParticipant] = [PAXParticipant(i, X[i]) for i in range(amount_participants)]
        A:PAXAggregator = PAXAggregator(P)
        self.A = A

        A.create_global_null_model(X[0].shape[1]) # line 1
        epsilonP:np.ndarray[float] = A.compute_local_epsilon(eA) # line 2 & 6

        for i in range(amount_participants): # line 4-8
            Pi:PAXParticipant = P[i]
            Pi.recieve_e(epsilonP[i]) # line 5
            Pi.compute_histogram() # line 7

        t = 0 # amount of trees counter
        trees = []
        while t <= T:
            DA, GA, HA = None # line 11 # TODO initialize properly
            for i in range(amount_participants):
                Pi:PAXParticipant = P[i]
                Pi.recieve_model(A.getmodel())
                Pi.predict()
                gpi, hpi = Pi.calculatedifferentials()
                DXpi = Pi.getDXpi()
                DA = None # line 17 # TODO take union
                GA = None # line 18 # TODO take union
                HA = None # line 19 # TODO take union

            emA = min(epsilonP) # line 21

            DmA, GmA, HmA = A.merge_hist(DA, GA, HA, emA) # line 25

            ftA = A.grow_tree(DmA, GmA, HmA)
            A.trees.append(ftA)

    def predict(self, X:np.ndarray):
        return self.A.predict()

class PAXAggregator:

    def __init__(self, P:list) -> None:
        self.P = P
        self.trees = []

    def create_global_null_model(self, numfeatures) -> TreeNode:
        self.trees.append(XGPyBoostClass.create_first_tree(numfeatures))

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

    def predict(self, X):
        pass

    def grow_tree(self, Dma, GmA, HmA):
        pass

    def merge_hist(self, DA, GA, HA, emA):
        pass

    def getmodel(self):
        return self.trees[-1]

class PAXParticipant:

    def __init__(self, id:int, X:np.ndarray) -> None:
        self.idk:int = id
        self.X:np.ndarray = X
        self.DXpi = None # local histogram
        self.e = None


    def getDXpi(self) -> object: # TODO change object to histogram
        return self.DXpi

    def compute_histogram(self) -> None: # TODO use self.X and self.e to construct local historgram on features. store it in self.DXpi
        self.sketch = DDSketch(self.e)
        for x in self.X:
            self.sketch.add(x)



    def calculatedifferentials(self) -> Tuple[np.ndarray, np.ndarray]: # use the prediction from self.prediction to calculate the differentials and store them in self.grad & self.hess
        pass

    def recieve_model(self, model):
        self.model = model

    def predict(self): # TODO predict using the model and local histogram
        # use self.model to predict
        pass

    def sendDataCount(self):
        return self.X.shape[0]

    def recieve_e(self, ep):
        self.e = ep