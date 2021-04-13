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

model = create_model(144, 128)
model.load_weights(r'test_data/station/s144 u128 f.h5')
scaler = pickle.load(open(r'test_data/station/s144 u128 f.p',"rb"))

folder = "test_data/station/station 2018 %anomaly 10% sigma 0.5 mode noise seed 109"
data1 = pd.read_csv(f"{folder}/anomaly.csv", encoding = "utf-8")

#############################################################################################

tmp_time = time.time()

options = {"target_col" : "water_lv",
          "DA_method" : "mad",
          "MW_size_DA" : 5,
          "addOri_to_DA" : False,
          "max_anomaly" : 24,
          "threshold" : 2.5,
          "min_mad" : 0.04,
          }

data_before, data_detected = anomaly_detection_online_LSTM(data1.iloc[:10000], model, step = 144, scaler = scaler, **options)

print("\nTotal time :", round(time.time()-tmp_time,4), "seconds")

score = cal_score(data_detected)
print(f"\n{score}")

#############################################################################################

fig1 = plot_type_1(data_before, df_new = data_detected, name = "MAD with LSTM online 1", new_water_lv = True, interval = True)
fig2 = plot_type_0(data_detected, name = "MAD with LSTM online 2")

fig1.write_image("plots/MAD with LSTM online 1.png")
fig2.write_image("plots/MAD with LSTM online 2.png")
fig1.write_html("plots/MAD with LSTM online 1.html")
fig2.write_html("plots/MAD with LSTM online 2.html")