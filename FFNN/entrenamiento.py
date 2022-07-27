# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:38:58 2022

@author: luisz
"""

import config
import pandas as pd
from binance.client import Client
import matplotlib.pyplot as plt
import os

cliente = Client(config.API_KEY, config.API_SECRET, tld='com')

#moneda
simbolo = 'BTCBUSD'

cierre=[] #4
# alto=[] #2
# bajo=[] #3

#Obtener datos 1000 minimo
def datos(vector,columna):
    #obtenemos el valor de los ultimos 1000 precios de btc
    klines = cliente.get_historical_klines(simbolo, Client.KLINE_INTERVAL_5MINUTE, "5000 minutes ago UTC") #5000 minutos dividido en 5 da 1000 unidades   
      
    #eje x
    for i in range (0,len(klines)):
            vector.append(float(klines[i][columna])) # 4 precio de cierre de la vela
        
    return vector

#cierre=datos(cierre,4)


import pickle
with open("lista de valores para entrenar.pickle", "rb") as f:
    cierre = pickle.load(f)

a=[]
for i in range(0,1000,1):
    a.append(cierre[i])

cierre=a


#####################################################################################################################################
#####################################           SET SEED TO ENSURE REPRODUCIBILITY      #############################################
seed_value=12345
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
import oil_static
from pandas import read_csv
## IMPORT MODULES AND DEFINE PATHS ##
import datetime


parent_folder='Control Time Series'
series_name='Precios de cierre de cripto'
output_variable='Precio de cierre'

##LOAD DATA##   
df = pd.Series (cierre)
df_univariate=df.to_frame()

##FEED FORWARD NEURAL NETWORK (FFNN)##
outsample= main_univariate.ffnn_univariate(
df_univariate, parent_folder=parent_folder, series_name=series_name, L=1, T=1,
ffnn_nodes1=50, ffnn_nodes2=2000, ffnn_epochs=250, ffnn_batch=300, ffnn_optimizer='nadam')


y=[]
for i in range(0,len(outsample),1):
    y.append(outsample.iloc[i,0])
  
    
with open("lista de valores para entrenar.pickle", "rb") as f:
    cierre = pickle.load(f)
a=[]
for i in range(0,1200,1):
    a.append(cierre[i])
cierre=a


b=len(y)
fig, ax = plt.subplots(1)
ax.plot(cierre[-400:-200], label='original', color='blue')
ax.plot(y, label='predictions', color='red')
ax.legend(loc='upper right')
ax.set_xlabel('Time',fontsize = 16)
ax.set_ylabel('FFNN',fontsize = 16)
plt.show()

fig, ax = plt.subplots(1)
ax.plot(cierre[-b:], label='original', color='blue')
ax.plot(y, label='predictions', color='red')
ax.legend(loc='upper right')
ax.set_xlabel('Time',fontsize = 16)
ax.set_ylabel('FFNN 200 a futuro',fontsize = 16)
plt.show()


## DLSTM
serie=read_csv('prueba1000.csv',header=0,index_col=0)
predictions= oil_static.run(serie)


fig, ax = plt.subplots(1)
ax.plot(cierre[-400:-200], label='original', color='blue')
ax.plot(predictions[-200:], label='predictions', color='red')
ax.legend(loc='upper right')
ax.set_xlabel('Time',fontsize = 16)
ax.set_ylabel('DLSTM',fontsize = 16)
plt.show()

fig, ax = plt.subplots(1)
ax.plot(cierre[-200:], label='original', color='blue')
ax.plot(predictions[-200:], label='predictions', color='red')
ax.legend(loc='upper right')
ax.set_xlabel('Time',fontsize = 16)
ax.set_ylabel('DLSTM 200 a futuro',fontsize = 16)
plt.show()






# 52.54 ffnn 
# 52.83 dlstm













    