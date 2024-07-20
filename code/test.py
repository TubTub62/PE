import numpy as np
from mp_preprocess import preprocess
from utils import m_identity, m_aa
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = "../data/ube4b/"
    dn = "ube4b"
    #data = "../data/gbd1/"
    #dn = "gbd1"
    x_train, x_test, y_train, y_test = preprocess(
        data, dn, num_processes=10)
    
