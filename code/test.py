import numpy as np
from itertools import product
from code.model_search.builder_core import *

fl = filter_layers([1,2,3,4,5], [512, 256, 128, 64, 32], [16, 12, 8, 4], 'relu')
dl = dense_layers([1,2,3,4], [256, 128, 64, 32], 'relu')
det = details([True, 0.25], [0.0001, 0.001, 0.01])
epochs = 2

ci = construction_info(fl, dl, det, epochs)

test = [fl.f_size for _ in fl.f_size]


f_size_product = product(fl.f_size, repeat=5)

res = product(fl.num_layers, fl.k_size, dl.num_layers)


options = 0
print(res)
    
print(options)