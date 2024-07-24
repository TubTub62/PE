import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os

def pca(path, n_components):

    df = pd.read_csv(path)
    df = df.drop('Index', axis=1)

    scaler = MinMaxScaler((0,1))
    scaler.fit(df)
    data = scaler.transform(df)

    pca = PCA(n_components=n_components)
    pca.fit(data)
    np.save(os.path.join("../data/aaindex","aaindex_pca.npy"), pca.components_)

if __name__ == "__main__":
    pca("../data/aaindex/AAindex1_fixed.csv", 5)