import pandas as pd
import numpy as np
from aaindex import aaindex1

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
    np.save("aaindex.npy", data)

if __name__ == '__main__':
    create_data()