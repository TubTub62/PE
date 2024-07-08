import pandas as pd
import numpy as np
from create_data_gb1 import create_xy
import tensorflow as tf
import keras
from keras.api.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.api.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
os.system('clear')

df = pd.read_excel("../data/gbd1_data.xlsx", header=2, usecols="B:K,N:R", nrows=200000)
f = open("../data/base_gbd1.txt")
base = f.readline()
f.close()
df = df.sample(frac=1)
os.system('clear')
print("Data Read")
print(df.head())

mutants, fitness = create_xy(df, base)
os.system('clear')
print("Split Into Mutants And Fitness")


def aa_identity(aa):
    aa_string = "ARNDCQEGHILKMFPSTWYV"
    for i in range(len(aa_string)):
        if aa == aa_string[i]:
            return i


def m_identity(mt):
    mi = np.zeros((20, len(mt)))
    for i in range(len(mt)):
        row = aa_identity(mt[i])
        mi[row][i] = 1
    return mi

mutant_identity = []
for mutant in mutants:
    mutant_identity.append(m_identity(mutant))
mutant_identity = np.array(mutant_identity)
print(mutant_identity.shape)
print(fitness.shape)
os.system('clear')
print("Created Identity Data")



x_train, x_test, y_train, y_test = train_test_split(mutant_identity, fitness, test_size=0.2, shuffle=True, random_state=15)
print(f"{x_train.shape}\n{x_test.shape}\n{y_train.shape}\n{y_test.shape}")
os.system('clear')
print("Split Data Into Training And Testing")
print("Running Model")

def model_1(x_train, y_train, x_test, y_test, epochs=20, batch_size=64):
    model = Sequential()
    model.add(keras.Input((20,56)))
    #model.add(Flatten())
    model.add(Conv1D(filters=128, kernel_size=12, activation='relu', use_bias=True, padding='same'))
    model.add(MaxPooling1D(2))    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', use_bias=True))
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu', use_bias=True))
    model.add(Flatten())
    model.add(Dense(128, activation='relu')) #evt tilføj dropout istedet for så mange dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # Change to the appropriate number of classes for multi-class classification
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))
    return model, history


model, history = model_1(x_train, y_train, x_test, y_test, epochs=5)

loss, accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
os.system('clear')
print("Test Accuracy:", accuracy)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
