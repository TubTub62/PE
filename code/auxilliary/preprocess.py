import multiprocessing as mp
from multiprocessing import Pool, freeze_support
import numpy as np
from sklearn.model_selection import train_test_split
from helper import *
from pca_aa import pca

def multip(num_processes, split_mutants):
    results = []
    processes = []
    identity = []
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    for i in range(num_processes):
        identity.append(i)
        process = ctx.Process(target=process_m_identity,
                   args=(queue, i, split_mutants[i]))
        processes.append(process)
        process.start()
    
    results = [queue.get() for p in processes]
    for p in processes:
        p.join()
    return results

def alternative(num_processes, split_mutants):
    input = []
    for i in range(num_processes):
        input.append((i, split_mutants[i]))
    with Pool(num_processes) as p:
        results = p.starmap(process_m_identity, input, 1)
    return results

def preprocess(num_processes=2, nrows=None, n_components=5):
    print("Importing Data")
    df = df_import("../data/gbd1_data.xlsx", nrows)
    base = import_base()

    clear()
    print("Creating Descriptors")
    mutants, fitness = create_xy(df, base)
    pca_aa = pca("../data/aa/AAindex1_fixed.csv", n_components)
    print("done pca and stuff")
    """ split_mutants = np.array_split(mutants, num_processes)
    results = alternative(num_processes, split_mutants)
    print(results)

    indexes = np.array([result[0] for result in results])
    unsorted_mutant_identities = np.array([result[1] for result in results])
    sorted_indexes = np.argsort(indexes)
    sorted_results = unsorted_mutant_identities[sorted_indexes]

    mutant_identity = np.row_stack(sorted_results)
    print(mutant_identity.shape) """

    mutant_identity = np.array([m_identity(mutant) for mutant in mutants])
    mutant_aa = np.array([m_aa(mutant, pca_aa, n_components) for mutant in mutants])
    print(mutant_aa.shape)
    mutant_stack = np.column_stack([mutant_identity,mutant_aa])

    #clear()
    print("Creating train/test splits")
    x_train, x_test, y_train, y_test = train_test_split(mutant_stack, fitness, test_size=0.2, shuffle=True, random_state=15)

    #clear()
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    freeze_support()