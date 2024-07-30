import numpy as np
from .builder_core import *
from .builder import builder
from itertools import product

def model_build(fl :filter_layers, dl : dense_layers, details : details,):

    models = []
    # Implement with itertools instead
    # https://docs.python.org/3/library/itertools.html#itertools.product
    
    combs = product(fl.num_layers, dl.num_layers, fl.k_size)
    
    for comb in combs:
        model = builder(comb[0], fl.f_size, comb[2], fl.activation,
                              comb[1], dl.d_size, dl.activation,
                              details.dropout[0], details.dropout[1])
        models.append(model)

    return models

def model_fit(models, epochs, x_train, y_train, x_test, y_test):

    print(f"Models to fit: {len(models)}")
    histories = []
    for i, model in enumerate(models):
        print(f"Fitting: {model}")
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=1,
                            validation_data=(x_test, y_test))
        histories.append(history)
    
    accuracies = []
    for model in models:
        loss, accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
        accuracies.append(accuracy)

    models = np.array(models)
    accuracies = np.array(accuracies)
    histories = np.array(histories)

    return accuracies, histories, models


def model_filter(accuracies, histories, models, num_best=5):

    sorted_indexes = np.argsort(accuracies)
    sort_acc = accuracies[sorted_indexes]
    sort_histories = histories[sorted_indexes]
    sort_models = models[sorted_indexes]

    sort_acc = sort_acc[::-1]
    sort_histories[::-1]
    sort_models = sort_models[::-1]

    sort_acc = sort_acc[:num_best]
    sort_histories[:num_best]
    sort_models = sort_models[:num_best]

    return sort_acc, sort_histories, sort_models

def model_save(models):

    for i, model in enumerate(models):
        model.save(f"model_{i}.keras", overwrite=True)

def model_summaries(models):
    
    for model in models:
        model.summary()
