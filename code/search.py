from model_search.builder_core import *
from model_search.model_search_core import *
from data_processing.mp_preprocess import preprocess


dpath = "data/gbd1/"
dname = "gbd1"

x_train, x_test, y_train, y_test = preprocess(dpath, dname, test_size=0.2, num_processes=10)

fl = filter_layers([1,2,3], [128, 64, 32], [4, 4, 4], 'relu')
dl = dense_layers([1,2,3], [128, 64, 32], 'relu')
aux = auxilliary([True, 0.25], [0.0001, 0.001, 0.01])


acc, hist, models = model_build_and_fit(fl, dl, aux,
                                        x_train, y_train, x_test, y_test,
                                        epochs=1)

acc, hist, models = model_filter(acc, hist, models, 5)


print(acc)

model_save(models)

model_summaries(models)