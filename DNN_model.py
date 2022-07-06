
import random
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.models import Sequential
from keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam

def set_seeds(seed = 100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def cw(df):
    c0, c1 = np.bincount(df["result"])
    w0 = (1/c0) * (len(df)) / 2
    w1 = (1/c1) * (len(df)) / 2
    return {0:w0, 1:w1}

optimizer = Adam(learning_rate = 0.0001)

def create_model(hl, hu, dropout, rate, regularize, reg = l2(0.0005), optimizer = optimizer, input_dim = None):
    if not regularize:
        reg = None
    model = Sequential()
    model.add(Dense(hu, input_dim = input_dim, activity_regularizer = reg ,activation = "relu"))
    #if dropout:  model.add(Dropout(rate, seed = 100))
    for layer in range(hl):
        model.add(Dense(hu, activation = "relu", activity_regularizer = reg))
        #if layer %2 == 0:  model.add(Dense(hu, activation="sigmoid", activity_regularizer=reg))
        if dropout and (layer % 3 == 0) :
            model.add(Flatten())
    #model.add(Dropout(0.1))
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
    #model.compile(loss = "mean_absolute_error", optimizer = optimizer, metrics = ["accuracy"])

    return model
