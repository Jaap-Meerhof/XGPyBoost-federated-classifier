DATA_PATH = "~/Documents/tmp/acquire-valued-shoppers-challenge/"
import pickle
import numpy as np
from sklearn.cluster import KMeans

def preprocess_purchase(self):
        IT_NUM = 600

        def populate():
            # Note: transactions.csv file can be downloaded from https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data
            fp = open(DATA_PATH+'transactions.csv')
            cnt, cust_cnt, it_cnt = 0, 0, 0
            items = dict()
            customer = dict()
            last_cust = ''
            for line in fp:
                cnt += 1
                if cnt == 1:
                    continue
                cust = line.split(',')[0]
                it = line.split(',')[3]
                if it not in items and it_cnt < IT_NUM:
                    items[it] = it_cnt
                    it_cnt += 1
                if cust not in customer:
                    customer[cust] = [0]*IT_NUM
                    cust_cnt += 1
                    last_cust = cust
                if cust_cnt > 250000:
                    break
                if it in items:
                    customer[cust][items[it]] = 1
                if cnt % 10000 == 0:
                    print(cnt, cust_cnt, it_cnt)
            del customer[last_cust]
            print(len(customer), len(items))

            no_purchase = []
            for key, val in customer.items():
                if 1 not in val:
                    no_purchase.append(key)
            for cus in no_purchase:
                del customer[cus]
            print(len(customer), len(items))
            pickle.dump([customer, items], open(DATA_PATH+'transactions_dump.p', 'wb'))

        populate()
        X = []
        customer, items = pickle.load(open(DATA_PATH+'transactions_dump.p', 'rb'))
        for key, val in customer.items():
            X.append(val)
        X = np.array(X)
        X = self.normalizeDataset(X)
        print(X.shape)
        pickle.dump(X, open(DATA_PATH+self.dataset_name+'_features.p', 'wb'))
        y = KMeans(n_clusters=100, random_state=0).fit(X).labels_
        pickle.dump(y, open(DATA_PATH+self.dataset_name+'_labels.p', 'wb'))
        print(np.unique(y))

preprocess_purchase()