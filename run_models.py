import joblib
import numpy as np
import pandas as pd 
import model_arima as ar 
import model_lstm as lstm 
import mosqlient as mosq
from epiweeks import Week
from scipy.stats import boxcox
from datetime import timedelta
from scipy.special import inv_boxcox

import matplotlib.pyplot as plt 

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("api_key")

def calcular_metricas_por_janela(array, tamanho_janela, funcoes):
    # Criar um array com as janelas deslizantes
    janelas = np.lib.stride_tricks.sliding_window_view(array, tamanho_janela)

    # Aplicar as funções de interesse em cada janela
    resultados = [func(janela, axis=0) for func in funcoes for janela in janelas]
    
    return np.array(resultados)

def get_slope(casos, axis = 0): 
     
    return np.polyfit(np.arange(0,4), casos, 1)[0]

# load data for a specific city
geocode = 3106200


df_c = mosq.get_infodengue(api_key = api_key,
                            disease = 'dengue', 
                          start_date = '2010-01-01',
                          end_date = '2025-11-10',
                          geocode = geocode)

df_c.data_iniSE = pd.to_datetime(df_c.data_iniSE)


# Range to train the models 
train_start_date = '2015-01-01'
train_end_date = '2024-12-22'

# arima 
df_ar = df_c[['data_iniSE', 'casos_est']].rename(columns = {'data_iniSE': 'dates', 'casos_est':'y'})

ar.train_model(df_ar, geocode, train_start_date = train_start_date, train_end_date = train_end_date)

# lstm 
df_lstm = df_c[['data_iniSE', 'casos_est', 'Rt', 'tempmed', 'umidmed']].rename(columns = {
    'data_iniSE': 'date', 
    'casos_est': 'casos'
})

df_lstm.set_index('date', inplace = True)

df_lstm['casos'] = boxcox(df_lstm.casos+1, lmbda=0.05)

df_lstm['SE'] = [Week.fromdate(x) for x in df_lstm.index]
    
df_lstm['SE'] = df_lstm['SE'].astype(str).str[-2:].astype(int)
    
df_lstm['SE'] = df_lstm['SE'].replace(53,52)
    
df_lstm['diff_casos'] = np.concatenate( (np.array([np.nan]), np.diff(df_lstm['casos'], 1)))
    
array = np.array(df_lstm.casos)
tamanho_janela = 4
    
df_lstm['casos_mean'] =  np.concatenate( (np.array([np.nan, np.nan, np.nan]), calcular_metricas_por_janela(array, tamanho_janela, [np.mean])))
    
df_lstm['casos_std'] =  np.concatenate( (np.array([np.nan, np.nan, np.nan]), calcular_metricas_por_janela(array, tamanho_janela, [np.std])))
    
df_lstm['casos_slope'] =  np.concatenate( (np.array([np.nan, np.nan, np.nan]), calcular_metricas_por_janela(array, tamanho_janela, [get_slope])))

df_lstm = df_lstm.dropna()

feat = df_lstm.shape[1]
HIDDEN = 64
LOOK_BACK = 4
PREDICT_N = 3
model_name = f'trained_{geocode}_dengue_city'

model = lstm.build_lstm(hidden=HIDDEN, features=feat, predict_n=PREDICT_N, look_back=LOOK_BACK,
                            batch_size=4, loss='mse')

model.compile(loss='mse', optimizer='adam', metrics=["accuracy", "mape", "mse"])


lstm.train_model(model,df_lstm, geocode, doenca='dengue',
                    end_train_date=None,
                    ratio = 1,
                    ini_date = train_start_date,
                    end_date = train_end_date,
                    min_delta=0.001, label='city',
                    patience = 30, 
                    epochs=300,
                    batch_size=4,
                    predict_n=PREDICT_N,
                    look_back=LOOK_BACK)


# apply the model 

for week in np.arange(1, 46):

    end_date = Week(2025, week).startdate() - timedelta(days = 7)
    end_date = end_date.strftime('%Y-%m-%d')

    df_apply = df_ar.loc[df_ar.dates <= pd.to_datetime(end_date)]

    df_for_arima = ar.apply_model(df_apply, geocode)

    df_for_lstm = lstm.apply_forecast(df_lstm, geocode, None, end_date = end_date, end_train_date = train_end_date, look_back=4, predict_n=3, model_name=model_name)










