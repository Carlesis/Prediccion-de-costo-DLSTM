# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:38:58 2022

@author: luisz
"""
seed_value=12345
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy
numpy.random.seed(seed_value)
import tensorflow
tensorflow.random.set_seed(seed_value)
from numba import jit, cuda

##FEED FORWARD NEURAL NETWORK (FFNN)##
def ffnn_univariate(df, parent_folder, series_name, L, ffnn_nodes1, ffnn_nodes2, ffnn_epochs, ffnn_batch, ffnn_optimizer, T=1):
    ##SET SEED TO ENSURE REPRODUCIBILITY##
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/prueba'
    parent_folder = pathlib.Path.home()/'Desktop/prueba' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/prueba' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/prueba' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/prueba' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_figures = pathlib.Path.home()/'Desktop/prueba' / parent_folder / series_name / 'Neural Networks/Univariate/Figures'
    univariate_ffnn = pathlib.Path.home()/'Desktop/prueba' / parent_folder / series_name / 'Neural Networks/Univariate/Figures/Feed-Forward Neural Network (FFNN)'
    univariate_tables = pathlib.Path.home()/'Desktop/prueba' / parent_folder / series_name / 'Neural Networks/Univariate/Tables'
    univariate_models= pathlib.Path.home()/'Desktop/prueba' / parent_folder / series_name / 'Neural Networks/Univariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_figures,univariate_ffnn,univariate_tables, univariate_models]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import pandas as pd
    import datetime
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        count=df.isnull().values.sum()
        if missing_values==True:
            print('The number of missing values is '+str(count))
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    ##FILL REMAINING MISSING VALUES WITH ZEROS##
    df=df.fillna(0)
    ##SPLIT TRAINING AND TEST SETS##
    def train_test (dataset, train):
        """Define and operate train-test split on dataset and return the length of the test set as scalar"""
        df=dataset.astype(float)
        train_size = int(len(df) * train)
        test_size = len(df) - train_size+1
        df_train, df_test = df[0:train_size], df[train_size:len(df)]
        print("Training Set observations: "+str(round(len(df_train)-len(df_train)*0.2))+" - Validation Set observations: "+str(round(len(df_train)*0.2))+" - Test Set observations: "+str(len(df_test)))
        return df_train, df_test, test_size
    df_train, df_test, test_size = train_test(df, train=0.8)
    ##DATA PREPARATION##
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set(font_scale=0.8)
    from keras.models import Sequential
    from keras.layers import SimpleRNN, LSTM, GRU, Dense
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.models import load_model
    from keras.regularizers import L1L2
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    from time import time
    ##PERFORMANCE METRICS##
    ##IMPORT MODULES##
    import statsmodels.tsa.api as smt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
    from prettytable import PrettyTable
    from pathlib import Path
    ##DEFINE METRICS##
    performance_metrics=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)', 'Mean Directional Accuracy (MDA)','R-Squared (R^2)']
    column_names=['Metric', 'Value']
    def performance_metrics_calculator(y_true,y_pred):
        """Compute metric scores for ML models"""
        mse=mean_squared_error(y_true,y_pred)
        rmse=sqrt(mse)
        mae=mean_absolute_error(y_true,y_pred)
        mape=mean_absolute_percentage_error(y_true,y_pred)
        mda=numpy.mean((numpy.sign(y_true[1:]-y_true[:-1])==numpy.sign(y_pred[1:]-y_pred[:-1])).astype(int))
        r2=r2_score(y_true, y_pred)
        metrics=[mse,rmse,mae,mape,mda,r2]
        return metrics
    ##CREATE TABLE##
    def performance_table(df, list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+str(df.columns[0])+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+str(df.columns[0])+" Series).txt").write_text(str(table))
        print(table)
        return table
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(df, actual, predicted, splits, path_name, model_name='', label='', color=''):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(df.columns[0])
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model ("+str(df.columns[0])+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##SCATTER PLOT PREDICTED VS. ACTUAL VALUES##
    def scatter_plot(actual, pred, path_name, model_name=''):
        fig=plt.figure()
        plt.scatter(actual, pred, color='steelblue')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted')
        ax = plt.gca()
        x = numpy.linspace(*ax.get_xlim())
        ax.plot(x, x, color='black', linestyle='dashed')
        plt.title(series_name+'\n'+model_name+' Scatter Plot')
        plt.tight_layout()
        plt.savefig(f"{path_name}/"+model_name+" Scatter Plot ("+series_name+").png", format="png")
        plt.close
    #NEURAL NETWORKS#
    ##TENSOR CREATION##
    def data_transform(data, L, T=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        var_names = data.columns.tolist()
        cols, names = list(), list()
        for i in range(L, 0, -1):
            cols.append(df.shift(i))
            names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
        for i in range(0, T):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
            else:
                names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    ##APPLY ALGORTIHM##
    T=1
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T:]
    train_x_data=data_train.iloc[: , :L]
    test_y_data=data_test.iloc[: , -T:]
    test_x_data=data_test.iloc[: , :L]
    ##SCALE DATA##
    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##SEED##
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    ##DEFINE THE FFNN MODEL##
    if len(df_train)>1000:
        ffnn_model = Sequential()
        ffnn_model.add(Dense(ffnn_nodes1, activation='relu'))
        ffnn_model.add(Dense(ffnn_nodes2, activation='relu'))
        ffnn_model.add(Dense(1))
        ffnn_model.compile(loss='mse', optimizer=ffnn_optimizer, metrics='mae')
    else:
        ffnn_model = Sequential()
        ffnn_model.add(Dense(ffnn_nodes1, activation='relu'))
        ffnn_model.add(Dense(1))
        ffnn_model.compile(loss='mse', optimizer=ffnn_optimizer, metrics='mae')
    ##SAVE BEST MODEL##
    ffnn_best_model=ModelCheckpoint(f"{univariate_models}/ffnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA AND RECORD TRAINING TIME##
    start=time()
    ffnn_fit=ffnn_model.fit(train_x, train_y, epochs=ffnn_epochs, batch_size=ffnn_batch, validation_split=0.2, verbose=2, callbacks=[ffnn_best_model])
    univariate_ffnn_training_time=time()-start
    ##EVALUATE FIT DURING TRAINING##
    def loss_plot(df, model,  path_name, name=''):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+str(df.columns[0])+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+str(df.columns[0])+" Series).png", format="png")
    loss_plot(df, ffnn_fit, path_name=univariate_ffnn, name="FFNN")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(df, model, path_name, name=''):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+str(df.columns[0])+" Series).png", format="png")
    mae_plot(df, ffnn_fit, path_name=univariate_ffnn, name='FFNN')
    ##LOAD BEST MODEL##
    ffnn_best_model=load_model(f"{univariate_models}/ffnn_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{univariate_models}/FFNN best model summary ("+series_name+").txt", 'w') as f:
        ffnn_best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    ##FIT MODEL ON TRAINING DATA##
    ffnn_train_predict=ffnn_best_model.predict(train_x,verbose=2)
    ##FIT MODEL ON TEST DATA##
    ffnn_test_predict=ffnn_best_model.predict(test_x,verbose=2)
    ##INVERT SCALING AND RESHAPE VECTORS##
    train_y_inverse = scaler_y.inverse_transform(train_y)
    test_y_inverse = scaler_y.inverse_transform(test_y)
    ffnn_train_predict_inv=scaler_y.inverse_transform(ffnn_train_predict)
    ffnn_test_predict_inv=scaler_y.inverse_transform(ffnn_test_predict)
    ##CREATE REFERENCE DATASETS##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    ffnn_train_predict_inv=pd.DataFrame(ffnn_train_predict_inv, index=df_train_for_plotting.index)
    ffnn_test_predict_inv=pd.DataFrame(ffnn_test_predict_inv, index=df_test_for_plotting.index)
    def forecast_error(y_true,y_pred):
        y_true=y_true.values
        y_pred=y_pred.values
        y_error=[y_true[i]-y_pred[i] for i in range(len(y_true))]
        return y_error
    ffnn_error_train = forecast_error(df_train_for_plotting,ffnn_train_predict_inv)
    ffnn_error_test = forecast_error(df_test_for_plotting, ffnn_test_predict_inv)
    ##COMPUTE PERFORMANCE METRICS##
    ffnn_metrics_train=performance_metrics_calculator(df_train_for_plotting.to_numpy(), ffnn_train_predict_inv.to_numpy())
    ffnn_metrics_test=performance_metrics_calculator(df_test_for_plotting.to_numpy(), ffnn_test_predict_inv.to_numpy())
    ##CREATE TABLES##
    ffnn_table_train=performance_table(df, ffnn_metrics_train,  path_name=univariate_tables, name='FFNN - In-Sample')
    ffnn_table_test=performance_table(df, ffnn_metrics_test,  path_name=univariate_tables, name='FFNN - Out-of-Sample')
    ##PLOTS (FULL)##
    plot_performance(df, df_train_for_plotting, ffnn_train_predict_inv, 1, path_name=univariate_ffnn, model_name='FFNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, ffnn_test_predict_inv, 1, path_name=univariate_ffnn, model_name='FFNN - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS##
    plot_performance(df, df_train_for_plotting, ffnn_train_predict_inv, 5, path_name=univariate_ffnn, model_name='FFNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, ffnn_test_predict_inv, 5, path_name=univariate_ffnn, model_name='FFNN - Out-of-Sample', label="Test", color='orangered')
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train_for_plotting, ffnn_train_predict_inv, path_name=univariate_ffnn, model_name='FFNN (Layers='+str(len(ffnn_model.layers))+') In-Sample')
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting, ffnn_test_predict_inv, path_name=univariate_ffnn, model_name='FFNN (Layers='+str(len(ffnn_model.layers))+') Out-of-Sample')
    plt.close('all')
    ##RETRIEVE NUMBER OF LAYERS##
    ffnn_layers = len(ffnn_model.layers)
    
    return ffnn_test_predict_inv
