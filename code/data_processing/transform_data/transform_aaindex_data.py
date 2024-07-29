import pandas as pd
import numpy as np
from aaindex import aaindex1
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os

def extract_records():
    record_names = aaindex1.record_codes()
    all_aa = []
    for aa in record_names:
        vals = aaindex1[aa].values
        t1 = vals.items()
        t2 = list(t1)
        all_aa.append(t2)
    return np.array(all_aa)

def create_data():
    data = extract_records()
    np.save(os.path.join("data/aaindex", "aaindex.npy"), data)

def aaindex_pca(path, n_components):

    df = pd.read_csv(path)
    df = df.drop('Index', axis=1)

    scaler = MinMaxScaler((0,1))
    scaler.fit(df)
    data = scaler.transform(df)

    pca = PCA(n_components=n_components)
    pca.fit(data)
    np.save(os.path.join("data/aaindex","aaindex_pca.npy"), pca.components_)

if __name__ == '__main__':
    create_data()
    aaindex_pca("data/aaindex/AAindex1_fixed.csv", 5)