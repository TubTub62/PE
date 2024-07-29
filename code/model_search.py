import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from builder_core import *

def model_builder(fl_n, f_size, k_size, f_act,
                  dl_n, d_size, d_act,
                  dp_b, dp_a):
    
    model = Sequential()

    # Introduce Convolution Part
    for i in range(fl_n):
        model.add(Conv1D(
            filters=f_size[i],
            kernel_size=k_size[i],
            activation=f_act
        ))

    if dp_b:
        model.add(Dropout(dp_a))

    # Dense Part
    model.add(Flatten())
    for i in range(dl_n):
        model.add(
            Dense(
                units=d_size[i],
                activation=d_act,
            )
        )
    model.add(Dense(2, activation='softmax'))

    #learning_rate = lr
    model.compile(
        optimizer=Adam(),
        loss="mse",
        metrics=["accuracy"]
    )
    return model

def model_search(fl :filter_layers, dl : dense_layers, aux : auxilliary,
                 x_train, y_train, x_test, y_test):

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
        history = model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1,
                            validation_data=(x_test, y_test))
        histories.append(history)
    
    accuracies = []
    for model in models:
        accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
        accuracies.append(accuracy)

    best_models = []
    models = np.array(models)
    accuracies = np.array(accuracies)
    sorted_indexes = np.argsort(accuracies)
    sort_acc = accuracies[sorted_indexes][-5:]
    sort_models = models[sorted_indexes][-5:]

    for i in range(5):
        print(sort_acc[i])
        sort_models[i].save(f"model_{i}.keras")

fl = filter_layers([1,2,3], [128, 64, 32], [4, 4, 4], 'relu')
dl = dense_layers([1,2,3], [128, 64, 32], 'relu')
aux = auxilliary([True, 0.25], [0.0001, 0.001, 0.01])