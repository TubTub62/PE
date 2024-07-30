import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import sys

def df_import(path, n=None):
    if n == None:
        return pd.read_excel(path, header=2, usecols="B:K,N:R")
    return pd.read_excel(path, header=2, usecols="B:K,N:R", nrows=n) 

def create_xy(df : pd.DataFrame, base):
    mutant_df = df[["Mut1 Position", "Mut1 Mutation", "Mut2 Position", "Mut2 Mutation"]]
    mutants = []
    rows, _ = df.shape
    for i in range(rows):
        pos1 = mutant_df.iloc[i].iloc[0]-1
        mut1 = mutant_df.iloc[i].iloc[1]
        pos2 = mutant_df.iloc[i].iloc[2]-1
        mut2 = mutant_df.iloc[i].iloc[3]
        wt1 = base[0:pos1] + mut1 + base[pos1+1:]
        wt2 = wt1[0:pos2] + mut2 + wt1[pos2+1:]
        mutants.append(wt2)

    scaler = MinMaxScaler((0,1))
    scaler.fit(df[["Mut1 Fitness", "Mut2 Fitness"]])

    fitness = scaler.transform(df[["Mut1 Fitness", "Mut2 Fitness"]])
    return np.array(mutants), fitness

if __name__ == "__main__":

    data_path = "data/gbd1/"

    if sys.argv[1] == "None":
        n = None
    else:
        n = int(sys.argv[1])
    
    df = df_import(data_path + "gbd1_data.xlsx", n)
    f = open(data_path + "base_gbd1.txt")
    base = f.readline()
    f.close()
    mutants, fitness = create_xy(df, base)

    np.save(os.path.join(data_path, "gbd1_mutants.npy"), mutants)
    np.save(os.path.join(data_path, "gbd1_fitness.npy"), fitness)