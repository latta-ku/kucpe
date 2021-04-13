# set_save_plot_show
# set_webgl_interactive_plot
# initial_display
# plot_water_lv
# save_df
# show_and_save_fig
# save_df
# select_file
# select_datetime
# select_detect_method
# select_misc_method
# rand_range

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import warnings
import traceback

from calendar import monthrange
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from .plots import *
from .model import *

save_plot_show = True
webgl_interactive_plot = True # if False will render plotly in svg mode

def set_save_plot_show(value):
    # REDACTED
    pass
def set_webgl_interactive_plot(value):
    # REDACTED
    pass

def initial_display():
    # REDACTED
    pass

######################################################################################################
######################################################################################################
######################################################################################################

def plot_water_lv(df, name = None, interactive_plot = False, webgl = webgl_interactive_plot):
    # REDACTED
    pass

def compare_2_dfs(df1, df2, name1 = None, name2 = None, same_plot = False, interactive_plot = False, webgl = webgl_interactive_plot):
    # REDACTED
    pass

######################################################################################################
######################################################################################################
######################################################################################################

def show_and_save_fig(fig, interactive = False):
    # REDACTED
    pass

######################################################################################################
######################################################################################################
######################################################################################################

def save_df(df):
    # REDACTED
    pass

######################################################################################################
######################################################################################################
######################################################################################################

def select_file(df, interactive_plot = False):
    # REDACTED
    pass

######################################################################################################
######################################################################################################
######################################################################################################
    
def select_datetime(df, df_output = None, interactive_plot= False):
    # REDACTED
    pass

######################################################################################################
######################################################################################################
######################################################################################################
    
def select_detect_method(df, df_output = None, interactive_plot = False):
    # REDACTED
    pass

######################################################################################################
######################################################################################################
######################################################################################################

def select_misc_method(df, df_output = None, interactive_plot = False):
    # REDACTED
    pass

######################################################################################################
######################################################################################################
######################################################################################################

def rand_range(start, end, K):

    # The origin of this function is https://www.geeksforgeeks.org/python-non-overlapping-random-ranges/

    if (sum(K) > end-start):
        print("K exceeds length of range")
        return

    ts = list(range(start,end))
    num = len(K)
    K = sorted(K)[::-1]
    skip = False
    tmplist = []

    for n in range(num):  
        while(True):
            temp = random.choice(ts)
            if (temp > end-K[n]):
                continue
            else:
                break

        ts2 = ts.copy()
        while any((temp >= tmplist[nn] and temp <= tmplist[nn] + K[nn] - 1) or \
                    ((temp + K[n] - 1) >= tmplist[nn] and (temp + K[n] - 1) <= tmplist[nn] + K[nn] - 1) or \
                    (temp > end-K[n]) \
                    for nn in range(len(tmplist))):
            
            ts2.remove(temp)
            if (len(ts2) == 0):
                print("Fragmentation")
                skip = True
                break
            temp = random.choice(ts2) 

        if (skip):
            break
        tmplist.append(temp)
        
        try:
            ts.remove(temp)
        except:
            pass
        try:
            ts.remove(temp + K[n] - 1)
        except:
            pass

    res = []
    for i in range(len(tmplist)):
        res.append([int(tmplist[i]), int(tmplist[i] + K[i] - 1)])

    return res