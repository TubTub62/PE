from multiprocessing import Pool
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *

def create_descriptors(mutants):
    pca_aa = np.load("data/aaindex/aaindex_pca.npy")

    mutant_identity = [m_identity(mutant) for mutant in mutants]
    mutant_aa = np.array([m_aa(mutant, pca_aa, pca_aa.shape[0]) for mutant in mutants])

    mutant_stack = np.column_stack([mutant_identity, mutant_aa])
    return mutant_stack


def preprocess_mp(num_processes, mutants):
    
    inp = []
    mutants_split = np.array_split(mutants, num_processes)
    for i in range(num_processes):
        inp.append((mutants_split[i]))


    clear()
    print("Pooling")
    with Pool(num_processes) as p:
        results = p.map(create_descriptors, inp, 1)

    return results

def preprocess(data_path, data_name, num_processes=2):
    print("Importing Data")
    mutants = np.load(data_path + f"{data_name}_mutants.npy")

    mutant_stacks = preprocess_mp(num_processes, mutants)
    
    clear()
    print("Consolidating & train/test Splitting")

    mutant_consol = np.row_stack(mutant_stacks)
    fitness = np.load(data_path + f"{data_name}_fitness.npy")

    x_train, x_test, y_train, y_test = train_test_split(
        mutant_consol, fitness, test_size=0.4, shuffle=True, random_state=133)
    
    return x_train, x_test, y_train, y_test
    

""" if __name__ == '__main__':
    freeze_support() """