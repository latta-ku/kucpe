import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.stats as stats
import tensorflow as tf
import os
import time
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Set seed
seed_value = 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# For tensorflow
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

##########################################################################################################################

def create_model(input_size, LSTM_unit = 128):
    model = tf.keras.models.Sequential()    
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_unit, input_shape=(input_size, 1))))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.build([1, input_size, 1])
    return model

# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def lstm(data, col, step, model, train_ratio, epochs, batch_size, monitor_matrix, patience, min_delta, reverse = False, random_test = False, verbose = 1):
    # REDACTED
    pass