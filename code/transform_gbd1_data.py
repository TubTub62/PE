import numpy as np
from utils import df_import, create_xy
import os

data_path = "data/gbd1/"

df = df_import(data_path + "gbd1_data.xlsx")
f = open(data_path + "base_gbd1.txt")
base = f.readline()
f.close()
mutants, fitness = create_xy(df, base)

np.save(os.path.join(data_path, "gbd1_mutants.npy"), mutants)
np.save(os.path.join(data_path, "gbd1_fitness.npy"), fitness)