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

from .train_lstm import *

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

# anomaly_detection_offline
# lstm_pred_offline
# anomaly_detection_naive_online
# anomaly_detection_online_LR
# anomaly_detection_online_LSTM
# LSTM_detection_online

##########################################################################################################################

# Calculate f1-score and other things
def cal_score(df):
    # df is a dataframe that has columns named anomaly_type and anomaly_type_label 

    # Define a dictionary for counting TP, FP, TN and FN
    count = {"TP" : 0,
             "FP" : 0,
             "TN" : 0,
             "FN" : 0
            }

    # Retrieve anomaly_type (values from anomaly detection processes not from labelling)
    anomaly_type = df["anomaly_type"].to_list()
    # Retrieve label
    anomaly_type_label = df["anomaly_type_label"].to_list()
    
    # Counting
    for index in range(len(df.index)):
        # Skip if NaN
        if (anomaly_type[index] == 2):
            continue
        # Negative cases
        elif (anomaly_type[index] != 1):
            # True negative
            if (anomaly_type_label[index] != 1):
                count["TN"] += 1
            # False negative
            else:
                count["FN"] += 1
        # Positive cases
        elif (anomaly_type[index] == 1):
            # True positive
            if (anomaly_type_label[index] == 1):
                count["TP"] += 1
            # False positive
            else:
                count["FP"] += 1

    # print(count)

    try:
        # Calculate rates
        TPR = count["TP"] / (count["TP"]+ count["FN"])
        FPR = count["FP"] / (count["TN"]+ count["FP"])
        TNR = count["TN"] / (count["TN"]+ count["FP"])
        FNR = count["FN"] / (count["TP"]+ count["FN"])
        # Calculate precision, recall and accuracy
        prec = count["TP"] / (count["TP"]+ count["FP"])
        recall = count["TP"] / (count["TP"]+ count["FN"])
        accuracy = (count["TP"] + count["TN"]) / len(df.index)
        # Calculate f1-score
        f1 = 2 * (prec * recall) / (prec + recall)
    except ZeroDivisionError:
        TPR = 0
        FPR = 0
        TNR = 0
        FNR = 0
        prec = 0
        recall = 0
        accuracy = 0
        f1 = 0
        print(f"\nZeroDivisionError\nF1-score can't be calculated\n{count}")
    
    return {"TPR" : TPR * 100,
            "FPR" : FPR* 100,
            "TNR" : TNR * 100,
            "FNR" : FNR * 100,
            "precision" : prec,
            "recall" : recall,
            "accuracy" : accuracy,
            "f1" : f1
            }

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
# Offline Detection

def med_fixed(x, y, threshold):
    # REDACTED
    pass

def mad(x, y, threshold, min_mad = 0.01):
    # Find mad
    med = np.median(x)
    mad = np.median(np.abs(np.array(x) - med))
    # If mad is less than min_mad
    # Let mad = min_mad
    if (mad < min_mad):
        mad = min_mad
    # Convert mad to madn
    madn = mad / 0.6745

    # Calculate z-score using madn
    z = abs(y - med) / madn
    # Check if z is more than threshold
    if (z <= threshold):
        # y is not an anomaly
        anomaly_type = 0
    else:
        # y is an anomaly
        anomaly_type = 1

    # Calculate inverval
    diff = threshold * madn
    upper = med + diff
    lower = med - diff

    return x, anomaly_type, upper, lower

def kmeans(x, y, threshold):
    # REDACTED
    pass

def anomaly_detection_offline(data, **kwargs):
    # REDACTED
    pass

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
# Offline Predivtion 

def find_nan_interval(df, col, min_rows, tolerance):
    
    interval = []
    x = []
    state = 0
    temp = min_rows
    tempz = tolerance
    
    for index, row in df.iterrows():
        if (state == 0 and np.isnan(row[col])):
            x.append(index)
            state = 1
            temp -= 1
        elif (state == 1 and np.isnan(row[col])):
            temp -= 1
            tempz = tolerance
        elif (state == 1 and not np.isnan(row[col])):
            if (temp <= 0 and tempz > 0):
                if (tempz == tolerance):
                    tempindex = index
                tempz -= 1
                continue
            elif (temp <= 0 and tempz <= 0):
                if (tolerance != 0):
                    x.append(tempindex-1)
                else:
                    x.append(index-1)
                interval.append(x)
                state = 0
                x = []
            else:
                state = 0
                x = []
                
            temp = min_rows
            tempz = tolerance
            
    if (state == 1):
        x.append(index)
        interval.append(x)
            
    return interval

def lstm_pred_offline(data, step, model, scaler = None, model_rev = None, scaler_rev = None):
    
    ###################################################

    # Predict forward
    water_lv_list = data["water_lv"].to_list()

    for index in range(0, len(water_lv_list)):

        # Skip if not NaN
        tmp_pass = False
        if (not np.isnan(water_lv_list[index])):
            tmp_pass = True

        if (not tmp_pass):
            # Create input
            pred_input = [water_lv_list[index - step : index]]
            # If use scaler
            if (scaler != None):
                pred_input = scaler.transform(pred_input)
            # Reshape to 3D [samples, timesteps, features]
            pred_input = np.array(pred_input, dtype=np.float32).reshape((1, step, 1))
            # Predict
            y = model(pred_input, training=False)
            y = y.numpy().flatten()[0]
            # If use sclaer  
            if (scaler != None):  
                y = scaler.inverse_transform([[y]]).flatten()[0]
            # Assign prediction to list
            water_lv_list[index] = y

        if (index % 100 == 0):
            print(f"[{index}/{len(water_lv_list)}]")
    
    print(f"[{len(water_lv_list)}/{len(water_lv_list)}]")

    # Assign to dataframe
    data["water_lv_forward"] = water_lv_list

    ###################################################

    # If predict backward
    if (model_rev != None):
        water_lv_list = data["water_lv"].to_list()[::-1]

        for index in range(0, len(water_lv_list)):

            # Skip if not NaN
            tmp_pass = False
            if (not np.isnan(water_lv_list[index])):
                tmp_pass = True
            
            if (not tmp_pass):
                # Create input
                pred_input = [water_lv_list[index - step : index]]
                # If use scaler
                if (scaler_rev != None):
                    pred_input = scaler_rev.transform(pred_input)
                # Reshape to 3D [samples, timesteps, features]
                pred_input = np.array(pred_input, dtype=np.float32).reshape((1, step, 1))
                # Predict
                y = model_rev(pred_input, training=False)
                y = y.numpy().flatten()[0]
                # If use sclaer  
                if (scaler_rev != None):  
                    y = scaler_rev.inverse_transform([[y]]).flatten()[0]
                # Assign prediction to list
                water_lv_list[index] = y

            if (index % 100 == 0):
                print(f"[{index}/{len(water_lv_list)}]")

        # Assign to dataframe
        data["water_lv_backward"] = water_lv_list[::-1]

        print(f"[{len(water_lv_list)}/{len(water_lv_list)}]")

        ###################################################

        # Find interval of NaN from the original data
        nan_interval_index = find_nan_interval(data, "water_lv", 1, 0)
        # Combine backward and forward predictions
        tmp = data["water_lv"].to_list()
        for nan_interval in nan_interval_index:
            # Number of predictions in this interval
            n_pred = nan_interval[1] - nan_interval[0] + 1
            # If there's only one prediction, let weight be 0.5
            if (n_pred == 1):
                weight = [0.5]
            # Create weights
            else:
                # Calculate step of weights
                step = 1 / (n_pred-1)
                # Create weights
                weight = np.arange(0, 1.2, step)[:n_pred]
            # Combine backward and forward values using weights
            for index in range(nan_interval[0], nan_interval[1] + 1):
                tmp[index] = (data["water_lv_forward"].iloc[index] * (1-weight[index - nan_interval[0]])) + (data["water_lv_backward"].iloc[index] * weight[index - nan_interval[0]])
        # Assign to dataframe
        data["water_lv_combined"] = tmp

        ###################################################

    return data

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
# Online Detection

def calculate_PI_naive(data, alpha):
    # REDACTED
    pass

def mad2(x, y, threshold, min_PI):
    # REDACTED
    pass

def anomaly_detection_naive_online(data, **kwargs):
    # REDACTED
    pass

def anomaly_detection_online_LR(data, **kwargs):
    # REDACTED
    pass

def anomaly_detection_online_LSTM(data, model, step, scaler = None, **kwargs):

    ###################################################

    # Default options 
    options = {"target_col" : "water_lv",
              "DA_method" : "kmeans",
              "MW_size_DA" : 20,
              "addOri_to_DA" : True,
              "max_anomaly" : None,
              "threshold" : None,
              "min_mad" : 0.01,
              "min_value" : None,
              "max_value" : None,
              "max_outofbound" : 1
              }
    # Update options
    options.update(kwargs)
    
    ##################################################

    # Copy main dataframe
    data_copy = data.copy()
    data_copy = data_copy.astype({options["target_col"] : float})
    # Reset index
    data_copy = data_copy.reset_index(drop = True)

    # Create output dataframe
    data_output = data_copy.copy()

    # Label NaN with 2 and 0 on anything else
    tmp = np.array([0] * len(data_copy.index))
    tmp[data_output.isna()[options["target_col"]]] = 2
    data_output["anomaly_type"] = tmp

    ##################################################
    
    # Retrieve anomaly_type and values
    anomaly_type_list = data_output["anomaly_type"].to_list()
    water_lv_list = data_output["water_lv"].to_list()
    # Create list for input that will be used to detect anomaly
    MW_DA_list = [None for i in range(len(data_output))]
    # Create list for upper and lower
    upper_list = [None for i in range(len(data_output))]
    lower_list = [None for i in range(len(data_output))]

    # Determine where to start
    start_index = max(options["MW_size_DA"] - 1, step)
    # limit is where moving window will stop
    limit = len(water_lv_list)
    # target_index is the position of the value in moving window
    # that will be determined if it's an anomaly or not.
    target_index = options["MW_size_DA"] - 1

    # Retrieve values for moving window for detection
    MW_DA = water_lv_list[start_index - (options["MW_size_DA"] - 1) : start_index + 1]

    # Define wait_len
    # wait_len is the number of values that will be skipped
    wait_len = 0
    # Define anomaly_count
    # If options["max_anomaly"] is not None
    # anomaly_count = options["max_anomaly"]
    # anonaly_count is the maximum amount of anomalies that can be detected consecutively
    # If anomaly_count is decreased to 0
    # [start_index] values will be skipped
    anomaly_count = None
    if (options["max_anomaly"] != None):
        anomaly_count = options["max_anomaly"]
    outofbound_count = options["max_outofbound"]

    ###################################################

    time_tmp = time.time()

    # loop form start_index to the end
    for index in range(start_index, len(water_lv_list)):

        if (not np.isnan(water_lv_list[index])):
            outofbound = False
            if (options["min_value"] != None):
                if (water_lv_list[index] < options["min_value"]):
                    outofbound = True
            if (options["max_value"] != None):
                if (water_lv_list[index] > options["max_value"]):
                    outofbound = True

            if (outofbound):
                outofbound_count -= 1
            else:
                outofbound_count = options["max_outofbound"]
                
            if (outofbound_count == 0):
                anomaly_type_list[index] = 1
                break

        # if wait or nan
        if (wait_len > 0 or np.isnan(water_lv_list[index])):
            # if wait
            if (wait_len > 0):
                wait_len -= 1

            # if nan predict and substitute nan
            if (np.isnan(water_lv_list[index])):
                # label NaN
                anomaly_type_list[index] = 2
                # Predict and substitute the anomaly with predicted value
                MW_PD = [water_lv_list[index - step : index]]
                if (scaler != None):
                    MW_PD = scaler.transform(MW_PD)
                MW_PD = np.array(MW_PD, dtype=np.float32).reshape((1, step, 1))
                y = model(MW_PD, training=False)
                y = y.numpy().flatten()[0]
                if (scaler != None):
                    y = scaler.inverse_transform([[y]]).flatten()[0]        
                water_lv_list[index] = y
                # if not add original value to moving window only 
                # add prediction to moving window
                if (not options["addOri_to_DA"]):
                    MW_DA = MW_DA[1:]
                    MW_DA.append(y)

            # if not nan do nothing
            else:
                # Label non anomaly
                anomaly_type_list[index] = 0
                # add value to moving window
                MW_DA = MW_DA[1:]
                MW_DA.append(water_lv_list[index])

        ###################################################

        else:

            # if not wait and not nan
            # append new value to mw
            MW_DA = MW_DA[1:]
            MW_DA.append(water_lv_list[index])
            # Determine if a value is an anomaly or not
            # anomaly_type = 0 is not an anomaly
            # anomaly_type = 1 is an anomaly
            if (options["DA_method"] == "med_fixed"):
                MW_DA_list[index], anomaly_type, upper, lower = med_fixed(MW_DA, MW_DA[target_index], options["threshold"])
                upper_list[index] = upper
                lower_list[index] = lower
            elif (options["DA_method"] == "mad"):
                MW_DA_list[index], anomaly_type, upper, lower = mad(MW_DA, MW_DA[target_index], options["threshold"], options["min_mad"])
                upper_list[index] = upper
                lower_list[index] = lower
            elif(options["DA_method"] == "kmeans"):
                MW_DA_list[index], anomaly_type = kmeans(MW_DA, MW_DA[target_index], options["threshold"])

            ###################################################

            # If not anomaly
            if (anomaly_type == 0):
                anomaly_type_list[index] = anomaly_type
                # Reset anomaly_count
                anomaly_count = options["max_anomaly"]
            # If anomaly
            elif (anomaly_type == 1):
                anomaly_type_list[index] = anomaly_type
                # If anomaly_count is set
                # Decrease it by one
                if (anomaly_count != None):
                    anomaly_count -= 1
                    # If anomaly_count reaches zero
                    # Reset anomaly_count
                    # Skip [start_index] values
                    if(anomaly_count == 0):
                        anomaly_count = options["max_anomaly"]
                        wait_len = start_index

                # Predict and substitute the anomaly with predicted value
                MW_PD = [water_lv_list[index - step : index]]
                if (scaler != None):
                    MW_PD = scaler.transform(MW_PD)
                MW_PD = np.array(MW_PD, dtype=np.float32).reshape((1, step, 1))
                y = model(MW_PD, training=False)
                y = y.numpy().flatten()[0]
                if (scaler != None):
                    y = scaler.inverse_transform([[y]]).flatten()[0]        
                water_lv_list[index] = y

                # if not add original value to moving window only 
                # replace the original value with the predicted value 
                if (not options["addOri_to_DA"]):
                    MW_DA[-1] = y

        # Output massage
        if (index % 100 == 0):
            try:
                pps = round(100/(time.time() - time_tmp),2)
            except:
                pps = 0
            print(f"[{index}/{len(water_lv_list)}]\tPoints per second : {pps}")
            time_tmp = time.time()

    # Output massage
    print(f"[{len(data.index)}/{len(data.index)}]")

    ###################################################

    # Assign values to output dataframe
    data_output["water_lv"] = water_lv_list
    data_output["anomaly_type"] = anomaly_type_list
    # data_output["MW_DA"] = MW_DA_list
    data_output["upper"] = upper_list
    data_output["lower"] = lower_list

    print("DONE")

    return data_copy, data_output

def LSTM_detection_online(data, model, step, scaler = None, **kwargs):
    # REDACTED
    pass
