import os
cwd = os.getcwd()
os.chdir(r'./..')

import time
import pickle
from function_files.functions import *
from function_files.train_lstm import *
from function_files.plots import *

os.chdir(cwd)

#############################################################################################

# Set seed for reproducible results
seed_value = 42
import os
import random
import numpy as np
import tensorflow as tf
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

#############################################################################################
    
def create_model(input_size, LSTM_unit = 128):
    model = tf.keras.models.Sequential()    
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_unit, input_shape=(input_size, 1))))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.build([1, input_size, 1])
    return model

#############################################################################################

# Create model for forward predictions
model = create_model(144, 128)
model.load_weights(r'test_data/station/s144 u128 f.h5')
scaler = pickle.load(open(r'test_data/station/s144 u128 f.p',"rb"))

# Create model for backward predictions
model_rev = create_model(144, 128)
model_rev.load_weights(r'test_data/station/s144 u128 b.h5')
scaler_rev = pickle.load(open(r'test_data/station/s144 u128 b.p',"rb"))

# Load data
folder = "test_data/station/station 2018 %anomaly 10% sigma 0.5 mode noise seed 109"
data = pd.read_csv(f"{folder}/non_anomaly.csv", encoding = "utf-8")
data_predict = data.copy()

#############################################################################################

# Random the parts of data we're deleting
# Define percent of data we're deleting
percent = 10
# Random the amount of intervals of deletion
a = np.random.randint(100, 200, size=1) 
b = np.ones(a)
# Random the size of each interval
K = list(  ( np.random.dirichlet(b, size=1)[0] * round(len(data_predict.index) * (percent/100)) ).round(0)  )
# Randomly place the intervals 
cut_range_list = rand_range(0, len(data_predict.index), K)

# Delete some parts of data
for cut_range in cut_range_list:
    data_predict["water_lv"].iloc[cut_range[0] : cut_range[1] + 1] = np.nan
    
# Make sure we don't have any NaNs in the first inputs
# Error will occur if there're NaNs in the inputs
data_predict["water_lv"].iloc[0:144] = data["water_lv"].iloc[0:144]
data_predict["water_lv"].iloc[-144:] = data["water_lv"].iloc[-144:]

#############################################################################################

tmp_time = time.time()

data_predict = lstm_pred_offline(data_predict, step = 144, model = model, scaler = scaler, model_rev = model_rev, scaler_rev = scaler_rev)

print("\nTotal time :", round(time.time()-tmp_time,4), "seconds")

#############################################################################################

fig = plot_type_5(data, data_predict, cut_range_list = cut_range_list, name = "LSTM prediction")

fig.write_image("plots/LSTM prediction.png")
fig.write_html("plots/LSTM prediction.html")