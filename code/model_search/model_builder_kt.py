import tensorflow as tf
import keras
from keras_tuner import HyperParameters, RandomSearch
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.optimizers import Adam
#import keras_tuner as kt


def model_builder(hp : HyperParameters):
    model = Sequential()

    # Introduce Convolution Part
    for i in range(hp.Int("num_con_layers"), 1, 3):
        model.add(Conv1D(
            filters=hp.Int(f"filter_layer_{i}", min=32, max=128, step=32),
            kernel_size=12-4*i,
            activation='relu'
        ))

    # Dense Part
    for i in range(hp.Int("num_dense_layers"), 1, 3):
        model.add(Flatten())
        model.add(
            Dense(
                units=hp.Int(f"units_dense_{i}", min=32, max=512, step=32),
                activation='relu',
            )
        )
    if hp.Boolean("dropout"):
        model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    learning_rate = hp.Float("learning_rate", min=0.0001, max=0.01, sampling="log")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["accuracy"]
    )
    return model

tuner = RandomSearch(
    hypermodel=model_builder,
    objective='val_accuracy',
    max_trials=2,
    max_consecutive_failed_trials=0,
)

print(tuner.search_space_summary())