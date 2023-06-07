
import numpy as np


def getDNA():
    import pandas as pd
    df=pd.read_csv("/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/dna.csv")
    print(df.head())
    labels = df.columns
    shape = df.shape
    y = df["class"]
    X = df.drop(['class'], axis=1)

    X = np.array(X)
    y = np.array(y)
    return X, y