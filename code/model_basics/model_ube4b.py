import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, MaxPooling2D
from keras.models import Sequential
from code.data_processing.utils import clear
from mp_preprocess import preprocess
from time import sleep

clear()
x_train, x_test, y_train, y_test = preprocess("data/ube4b/", "ube4b", num_processes=10)
print(f"trainshape:{x_train.shape}\ntestshape:{y_train.shape}")
print(y_test[0:10,:])

def model_ube4b(x_train, y_train, x_test, y_test, epochs=20, batch_size=64):
    model = Sequential()
    model.add(keras.Input((25,101)))
    model.add(Conv1D(filters=128, kernel_size=12, activation='relu', use_bias=True, padding='same'))
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', use_bias=True))
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu', use_bias=True))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu')) 
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax')) 
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))
    return model, history


model, history = model_ube4b(x_train, y_train, x_test, y_test, epochs=5)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=1)

clear()
print("Test Accuracy:", accuracy)

plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
