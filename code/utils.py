import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def df_import(path, n=None):
    if n == None:
        return pd.read_excel(path, header=2, usecols="B:K,N:R")
    return pd.read_excel(path, header=2, usecols="B:K,N:R", nrows=n) 
    

def import_base():
    f = open("../data/gbd1/base_gbd1.txt")
    base = f.readline()
    f.close()
    return base

def clear():
    os.system("clear")

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

def aa_identity(aa):
    aa_string = "ACDEFGHIKLMNPQRSTVWY"
    for i in range(len(aa_string)):
        if aa == aa_string[i]:
            return i


def m_identity(mt):
    mi = np.zeros((20, len(mt)))
    for i in range(len(mt)):
        row = aa_identity(mt[i])
        mi[row][i] = 1
    return mi

def m_aa(mt, aa, aa_n_components):
    mi = np.zeros((aa_n_components, len(mt)))
    for i in range(len(mt)):
        col = aa_identity(mt[i])
        for j in range(aa_n_components):
            mi[j][i] = aa[j][col]
    return mi
