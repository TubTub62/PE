import numpy as np
from .builder_core import *
from .model_builder import model_builder

def model_build_and_fit(fl :filter_layers, dl : dense_layers, aux : auxilliary,
                 x_train, y_train, x_test, y_test,
                 epochs=5):

    models = []
    fl_len = len(fl.num_layers)
    dl_len = len(fl.num_layers)
    for i in range(fl_len):
        for j in range(dl_len):
            model = model_builder(fl.num_layers[i], fl.f_size, fl.k_size, fl.activation,
                                dl.num_layers[j], dl.d_size, dl.activation,
                                aux.dropout[0], aux.dropout[1])
            models.append(model)

    for i in reversed(range(fl_len)):
        for j in reversed(range(dl_len)):
            model = model_builder(fl.num_layers[i], fl.f_size, fl.k_size, fl.activation,
                                dl.num_layers[j], dl.d_size, dl.activation,
                                aux.dropout[0], aux.dropout[1])
            models.append(model)
    histories = []
    for model in models:
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
