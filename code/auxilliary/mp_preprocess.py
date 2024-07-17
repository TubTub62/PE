from multiprocessing import Pool, freeze_support
import numpy as np
from sklearn.model_selection import train_test_split
from helper import *
from pca_aa import pca

def create_descriptors(df, n_components):
    base = import_base()

    mutants, fitness = create_xy(df, base)
    pca_aa = pca("../data/aa/AAindex1_fixed.csv", n_components)

    mutant_identity = [m_identity(mutant) for mutant in mutants]
    mutant_aa = np.array([m_aa(mutant, pca_aa, n_components) for mutant in mutants])

    mutant_stack = np.column_stack([mutant_identity, mutant_aa])

    return mutant_stack, fitness


def preprocess_helper(num_processes, df, n_components):
    
    inp = []
    df_split = np.array_split(df, num_processes)
    for df in df_split:
        inp.append((df, n_components))
    clear()
    print("Pooling")
    with Pool(num_processes) as p:
        results = p.starmap(create_descriptors, inp, 1)
    mt = []
    ft = []
    for res in results:
        mt.append(res[0])
        ft.append(res[1])
    return mt, ft

def preprocess(num_processes=2, nrows=None, n_components=5):
    print("Importing Data")
    df = df_import("../data/gbd1_data.xlsx", nrows)

    clear()
    mutant_stacks, fitness_stack = preprocess_helper(num_processes, df, 5)
    clear()
    print("Consolidating & train/test Splitting")

    mutant_consol = np.row_stack(mutant_stacks)
    fitness_consol = np.row_stack(fitness_stack)

    x_train, x_test, y_train, y_test = train_test_split(
        mutant_consol, fitness_consol, test_size=0.2, shuffle=True, random_state=15)
    
    return x_train, x_test, y_train, y_test
    

""" if __name__ == '__main__':
    freeze_support() """