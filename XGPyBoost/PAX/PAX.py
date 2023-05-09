import numpy as np
from XGPyBoostClass import *
from customddsketch import DDSketch
from typing import Tuple
from treemodel import TreeNode
from params import Params
from tqdm import tqdm

from histograms import histogram

class PAX:
    def __init__(self, model:XGPyBoostClass) -> None:
        self.model = model

    def fit(self, X:np.ndarray, y:np.ndarray, eA:float, T:int, l:callable, number_of_bins:int, splits:list[list[float]]=None) -> None:
        """Fit the model in a federated fashion

        Args:
            self (PAX) : PAX instance
            X (np.ndarray): array of data np.ndarrays. X[0] being the dataset of node 0
            y (np.ndarray): array of target np.ndarrays. y[0] being the target dataset of node 0
            eA (float): Global Error Tolence Budget
            T (int): Maximum Number of Training Rounds
            l (callable): Model Loss Function
        """
        n_classes = np.amax(y[0]) + 1

        amount_participants = len(X)
        P:list[PAXParticipant] = [PAXParticipant(i, X[i], y[i], T) for i in range(amount_participants)]
        A:PAXAggregator = PAXAggregator(P, n_classes=n_classes, model=self.model, number_of_bins=number_of_bins)
        self.A = A

        # A.create_global_null_model(X[0].shape[0]) # line 1
        A.create_global_null_model_known_probas(X[0].shape[0], y)
        epsilonP:np.ndarray[float] = A.compute_local_epsilon(eA) # line 2 & 6

        for i in range(amount_participants): # line 4-8
            Pi:PAXParticipant = P[i]
            Pi.recieve_e(epsilonP[i]) # line 5
            # Pi.compute_histogram(number_of_bins=number_of_bins) # line 7
            Pi.splits = splits


        t = 0 # amount of trees counter

        # self.trees = [[] for i in range(n_classes)]

        for t in tqdm(range(1, T), desc="building trees"):
        # for t in range(1, T):
            DA = [] # line 11 # TODO initialize properly
            GA = [] # line 11 # TODO initialize properly
            HA = [] # line 11 # TODO initialize properly
            for i in range(amount_participants):
                Pi:PAXParticipant = P[i]
                Pi.recieve_model(A.getmodel())
                Pi.predict()
                gpi, hpi = Pi.calculatedifferentials(l)
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

class PAXAggregator:

    def __init__(self, P:list, n_classes, model:XGPyBoostClass, number_of_bins:int) -> None:
        self.P = P
        self.n_classes = n_classes
        self.model = model
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

    def predict(self, X):
        probas = np.zeros((X.shape[0], self.n_classes, self.model.params.n_trees))

        y_pred = np.zeros((X.shape[0], self.n_classes, self.model.params.n_trees))

        for c in range(self.n_classes):
            for i, tree in enumerate(self.trees[c]):
                y_pred[:, c, i] = tree.predict(X)



        # now y_preds are filled
        for i in range(self.model.params.n_trees):
            for rowid in range(y_pred.shape[0]):
                row = y_pred[rowid, : , i]
                wmax = max(row) # line 100 multiclass_obj.cu
                wsum =0.0
                for y in row : wsum +=  np.exp(y - wmax)
                probas[rowid,:, i] = np.exp(row -wmax) / wsum

        probas = np.average(probas, axis=2)

        # tmp = [np.exp(x)/sum(np.exp(tmp)) for x in tmp]
        # binary = np.where(p >= 0.5, 1, 0)
        # binary_predictions[:, i+1] = binary
        # binary_predictions[:, i+1]

        # for i in range(X.shape[0]):
        #     average = sum(probas[i, :])/len(probas[i, :])
        #     probas[i, c] =  average

        # TODO take the highest probability and return its location in the list
        pass
            # for i in range(X.shape[0]):
            #     votes[i][c] = np.argmax(np.bincount(binary_predictions[i, :]))

        highets_prob = np.zeros((X.shape[0], self.n_classes), dtype='int64')
        for i in range(X.shape[0]):
            highets_prob[i] = np.where(probas[i] == np.amax(probas[i]), 1, 0)

        return [ np.argmax(probdrow ) for probdrow in highets_prob ]

    def grow_tree(self, Dma, GmA, HmA):

        def fit_tree_thread(c):
            tree = XGPyBoostClass._fit_tree(Dma, GmA[:, c], HmA[:, c], self.model.params)
            self.trees[c].append(tree)

        threads = []

        for c in range(self.n_classes):
            t:Thread = Thread(target=fit_tree_thread, args=(c,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

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

        # tmpDA = np.array(DA)

        # for feature in range(DA[0].shape[1]):
        #     unique_elements, indices, count = np.unique(tmpDA[:,:,feature].flatten(), return_index=True, return_counts=True)
        #     DMA[feature] = unique_elements

        #     for element in unique_elements:
        #         indices = np.where(tmpDA[:,:,feature].flatten() == element)
        #         weight = len(indices)
        #         tmpGA = np.array(GA)
        #         gradients = tmpGA[:,:,feature].flatten()[indices]
        #         gma = sum(gradients)/len(gradients)
        #         tmpHA = np.array(HA)
        #         hessians = tmpHA[:,:,feature].flatten()[indices]
        #         hma = sum(hessians)/len(hessians)
        #         DMA.append(element)
        #         GMA.append(gma)
        #         HMA.append(hma)
        #         pass
        #     pass


        # for pi in range(n_participants):
        #     pass
        #     DMA = DMA + DA[pi]
        # DMA = DMA/n_participants

        # for pi in range(n_participants):
        #     GMA = GMA + GA[pi]
        # GMA = GMA/n_participants

        # for pi in range(n_participants):
        #     HMA = HMA + HA[pi]
        # HMA = HMA/n_participants

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

    def compute_histogram(self, number_of_bins:int) -> None: # TODO use self.X and self.e to construct local historgram on features. store it in self.DXpi
        print("creating histogram for party {} 📊".format(self.idk))

        self.sketch = []
        self.histo = np.zeros((self.X.shape[1], number_of_bins))
        for feature in range(self.X.shape[1]):
            self.sketch.append(DDSketch(self.e))
            for x in self.X[:,feature]:
                self.sketch[feature].add(x)

            for i in range(1, number_of_bins+1):
                self.histo[feature][i-1] = self.sketch[feature].get_quantile_value(i/number_of_bins)




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
                split_indices = np.searchsorted(self.splits[feature_i][:-1], self.X[:, feature_i], side='left') # leave out last quantile
                interp_values[:, feature_i] = [(self.splits[feature_i][i-1] + self.splits[feature_i][i])/2 for i in split_indices ] # dit slaat nergens op !!!!
            self.DXpi = interp_values

        for class_i in range(self.n_classes):
            self.prediction[class_i] = self.model[class_i].predict(self.DXpi) # np.array(interp_values).T
        pass # DEBUG


    def sendDataCount(self):
        return self.X.shape[0]

    def recieve_e(self, ep):
        self.e = ep