import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from keras.models import Sequential
from utils import clear
from mp_preprocess import preprocess

def make_model():
    return