import pandas as pd
import numpy as np
from aaindex import aaindex1

def extract_records():
    record_names = aaindex1.record_codes()
    all_aa = []
    first = True
    for aa in record_names:
        vals = aaindex1[aa].values
        t1 = vals.items()
        t2 = list(t1)
        all_aa.append(t2)
    return np.array(all_aa)

def convert_to_pd(data):
    col_names = data[0][1:, 0]
    data_rows = []
    for entry in data:
        if entry[0][0] == '-':
            data_rows.append(entry[1:, 1])
        else:
            data_rows.append(entry[:][1])
    return pd.DataFrame(data=data_rows,columns=col_names)

def create_data():
    data = extract_records()
    res = convert_to_pd(data)
    res.to_csv("AAindex1_fixed.csv")

create_data()