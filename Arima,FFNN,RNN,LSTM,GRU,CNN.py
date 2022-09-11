# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:38:58 2022

@author: luisz
"""

import pandas as pd
from binance.client import Client
import matplotlib.pyplot as ptl
import numpy as np
import os
import pickle

with open("lista de valores para entrenar.pickle", "rb") as f:
    cierre = pickle.load(f)
print(len(cierre))

a=[]
for i in range (5000,8000,1):
    a.append(cierre[i])

#####################################################################################################################################
#####################################           SET SEED TO ENSURE REPRODUCIBILITY      #############################################
seed_value=12345
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy
numpy.random.seed(seed_value)
import tensorflow
tensorflow.random.set_seed(seed_value)
#####################################################################################################################################
###############################################         IMPORT MAIN SCRIPTS             #############################################
import main_univariate

## IMPORT MODULES AND DEFINE PATHS ##
import pandas as pd
import datetime
parent_folder='Control Time Series'
series_name='Precios de cierre de cripto3CNN'
output_variable='Precio de cierre'
##LOAD DATA## 

df = pd.Series (a)
#####################################################################################################################################
######################################                UNIVARIATE ANALYSIS               #############################################
#####################################################################################################################################
df_univariate=df.to_frame()


###################################     PRODUCE UNIVARIATE OUTPUT (OPTIMIZED PARAMETERS)      #######################################
#####################################################################################################################################
# ##DATA ANALYSIS##
main_univariate.data_analysis_univariate(df_univariate, parent_folder=parent_folder, series_name=series_name)
# ##ARIMA##
# arima_error_train_univariate, arima_error_test_univariate,arima_forecasts_test = main_univariate.arima(
# df_univariate, parent_folder=parent_folder, series_name=series_name)
##FEED FORWARD NEURAL NETWORK (FFNN)##
# univariate_ffnn_training_time, ffnn_metrics_train, ffnn_metrics_test, ffnn_layers, ffnn_error_train_univariate, ffnn_error_test_univariate = main_univariate.ffnn_univariate(
# df_univariate, parent_folder=parent_folder, series_name=series_name, L=1, T=1,
# ffnn_nodes1=50, ffnn_nodes2=2000, ffnn_epochs=250, ffnn_batch=300, ffnn_optimizer='nadam')
##RECURRENT NEURAL NETWORK (RNN)##
# univariate_rnn_training_time, rnn_metrics_train, rnn_metrics_test, rnn_layers, rnn_error_train_univariate, rnn_error_test_univariate = main_univariate.rnn_univariate(
# df_univariate, parent_folder=parent_folder, series_name=series_name, L=1, T=1,
# rnn_nodes1=200, rnn_nodes2=133, rnn_epochs=250, rnn_batch=43, rnn_optimizer='adam')
#LONG SHORT-TERM MEMORY (LSTM)##
# univariate_lstm_training_time, lstm_metrics_train, lstm_metrics_test, lstm_layers, lstm_error_train_univariate, lstm_error_test_univariate, lstm_test_predict_inv = main_univariate.lstm_univariate(
# df_univariate, parent_folder=parent_folder, series_name=series_name, L=1, T=1, 
# lstm_nodes1=400, lstm_nodes2=300, lstm_epochs=250, lstm_batch=30, lstm_optimizer='nadam')
# #GATED RECURRENT UNIT (GRU)##
# univariate_gru_training_time, gru_metrics_train, gru_metrics_test, gru_layers, gru_error_train_univariate, gru_error_test_univariate = main_univariate.gru_univariate(
# df_univariate, parent_folder=parent_folder, series_name=series_name, L=1, T=1,
# gru_nodes1=150, gru_nodes2=150, gru_epochs=250, gru_batch=150, gru_optimizer='adam')
# ##CONVOLUTIONAL NEURAL NETWORK (CNN)##
univariate_cnn_training_time, cnn_metrics_train, cnn_metrics_test, cnn_layers, cnn_error_train_univariate, cnn_error_test_univariate = main_univariate.cnn_univariate(
df_univariate, parent_folder=parent_folder, series_name=series_name, L=30, T=1,
cnn_filters_1=32, cnn_filters_2=32,cnn_kernel_1=2,cnn_kernel_2=2,cnn_dense_nodes=50,cnn_pool_size=2,cnn_epochs=250,cnn_batch=128,
cnn_optimizer='adam')


