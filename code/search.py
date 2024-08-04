from code.model_search.builder_core import *
from code.model_search.model_search_core import *
from code.data_processing.mp_preprocess import preprocess

def search_model(dname, ci : construction_info, filter_models=True, filter_amount=5, save_models=False, display_summaries=False):

    fl = ci.filter_layers
    dl = ci.dense_layers
    det = ci.details
    epochs = ci.epochs

    models, combs = model_build(fl, dl, det)

    x_train, x_test, y_train, y_test = preprocess(f"data/{dname}/", dname, test_size=0.2, num_processes=10)

    acc, hist, models = model_fit(models, combs, epochs, x_train, y_train, x_test, y_test)

    if filter_models:
        acc, hist, models = model_filter(acc, hist, models, num_best=filter_amount)

    if save_models:
        model_save(models)
    
    if display_summaries:
        model_summaries(models)
        print(acc)


fl = filter_layers([1,2,3,4], [256, 128, 64, 32], [16, 12, 8, 4], 'relu')
dl = dense_layers([1,2,3,4], [256, 128, 64, 32], 'relu')
det = details([True, 0.25], [0.0001, 0.001, 0.01])
epochs = 5

ci = construction_info(fl, dl, det, epochs)


search_model('gbd1', ci, True, 10, True, True)