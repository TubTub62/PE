import numpy as np
from utils import import_base, df_import, create_xy
import os

df = df_import("../data/gbd1/gbd1_data.xlsx", 5000)
base = import_base()
mutants, fitness = create_xy(df, base)

np.save(os.path.join("../data/gbd1", "gbd1_mutants.npy"), mutants)
np.save(os.path.join("../data/gbd1", "gbd1_fitness.npy"), fitness)