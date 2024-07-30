import numpy as np
from .builder_core import *
from .model_builder import model_builder

def model_build_and_fit(fl :filter_layers, dl : dense_layers, details : details,
                 x_train, y_train, x_test, y_test,
                 epochs=5):

    models = []
    
    fl_len = len(fl.num_layers)
    dl_len = len(dl.num_layers)
    k_len = len(fl.k_size)
    
    for i in range(fl_len):
        for j in range(dl_len):
            for k in range(k_len):
                model = model_builder(fl.num_layers[i], fl.f_size, fl.k_size[k], fl.activation,
                                    dl.num_layers[j], dl.d_size, dl.activation,
                                    details.dropout[0], details.dropout[1])
                models.append(model)

    for i in reversed(range(fl_len)):
        for j in reversed(range(dl_len)):
                for k in reversed(range(k_len)):
                    model = model_builder(fl.num_layers[i], fl.f_size, fl.k_size[k], fl.activation,
                                        dl.num_layers[j], dl.d_size, dl.activation,
                                        details.dropout[0], details.dropout[1])
                    models.append(model)

    print(f"Models to train: {len(models)}")

    histories = []
    for i, model in enumerate(models):
        print(f"Training model_{i}")
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
