import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys

if __name__ == "__main__":

    if sys.argv[1] == "None":
        n = None
    else:
        n = int(sys.argv[1])
    
    data_path = "data/ube4b/"

    f = open(data_path + "ube4b_small_base.txt")
    base = f.readline()
    f.close()

    df = pd.read_excel(data_path + "ube4b_fixed.ods", nrows=n)

    mutant_info = df['seqID'].to_numpy()

    wts = []
    for i in range(mutant_info.shape[0]):
        first_split = mutant_info[i].split("-")
        positions = first_split[0].split(",")
        positions = [int(i) for i in positions]
        aa_changes = first_split[1].split(",")
        wt = base
        for j in range(len(positions)):
            if positions[j] == 0:
                wt = aa_changes[j] + wt[positions[j]+1:]
            else:
                wt = wt[0:positions[j]-1] + aa_changes[j] + wt[positions[j]:]
        if len(wt) != 101:
            print(i)
            print(len(wt))
            print(mutant_info[i])
        wts.append(wt)

    wts = np.array(wts)
    np.save(os.path.join(data_path, "ube4b_mutants.npy"), wts)

    fitness = df[["log2_ratio", "nscor_log2_ratio"]]

    scaler = MinMaxScaler((0,1))
    scaler.fit(fitness)

    fitness = scaler.transform(fitness)

    np.save(os.path.join(data_path, "ube4b_fitness.npy"), fitness)