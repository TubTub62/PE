import pandas as pd
import numpy as np

def base_string(df):
    bs = "M"
    for i in range(2,56+1):
        t = df[df["Position"] == i]
        t = t["WT amino acid"].to_numpy()
        t = t[0]
        bs += t
    return bs

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
        #print(f"({pos1+1}, {mut1}), ({pos2+1}, {mut2}) : {wt2}")
        mutants.append(wt2)
    
    fitness_df = df[["Mut1 Fitness", "Mut2 Fitness"]].to_numpy()
    return np.array(mutants), fitness_df