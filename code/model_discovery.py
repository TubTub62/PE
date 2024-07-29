from builder_core import *
from model_search import model_search, model_summaries
from mp_preprocess import preprocess

dpath = "../data/gbd1/"
dname = "gbd1"

#x_train, x_test, y_train, y_test = preprocess(dpath, dname, 8)

""" fl = filter_layers([1,2,3], [128, 64, 32], [4, 4, 4], 'relu')
dl = dense_layers([1,2,3], [128, 64, 32], 'relu')
aux = auxilliary([True, 0.25], [0.0001, 0.001, 0.01]) """

fl = filter_layers([1,2,3], [128, 64, 32], [4, 4, 4], 'relu')
dl = dense_layers([1,2,3], [128, 64, 32], 'relu')
aux = auxilliary([True, 0.25], [0.0001, 0.001, 0.01])


""" model_search(fl, dl, aux,
             x_train, y_train, x_test, y_test,
             epochs=2, num_best=5) """

model_summaries(5)