import joblib
import pickle
import numpy as np
import pandas as pd
from epiweeks import Week
from datetime import datetime, timedelta
from pmdarima import preprocessing as ppc
from mosqlient.forecast import Arima

def get_next_n_weeks(ini_date: str, next_days: int) -> list:
    """
    Return a list of dates with the {next_days} days after ini_date.
    This function was designed to generate the dates of the forecast
    models.
    Parameters
    ----------
    ini_date : str
        Initial date.
    next_days : int
        Number of days to be included in the list after the date in
        ini_date.
    Returns
    -------
    list
        A list with the dates computed.
    """

    next_dates = []

    a = datetime.strptime(ini_date, "%Y-%m-%d")

    for i in np.arange(1, next_days + 1):
        d_i = datetime.strftime(a + timedelta(days=int(i * 7)), "%Y-%m-%d")

        next_dates.append(datetime.strptime(d_i, "%Y-%m-%d").date())

    return next_dates
    
def get_prediction_dataframe(preds_50, preds_80, preds_90, preds_95, date, boxcox) -> pd.DataFrame:
    """
    Function to organize the predictions of the ARIMA model in a pandas DataFrame.

    Parameters
    ----------
    horizon: int
        The number of weeks forecasted by the model
    end_date: str
        Last week of the out of the sample evaluation. The first week is after the last training observation.
    plot: bool
        If true the plot of the model out of the sample is returned
    """

    df_preds = pd.DataFrame()

    df_preds["date"] = date

    try:
        df_preds["pred"] = preds_95[0].values

    except:
        df_preds["pred"] = preds_95[0]

    df_preds.loc[:, ["lower_50", "upper_50"]] = preds_50[1]
    df_preds.loc[:, ["lower_80", "upper_80"]] = preds_80[1]
    df_preds.loc[:, ["lower_90", "upper_90"]] = preds_90[1]
    df_preds.loc[:, ["lower_95", "upper_95"]] = preds_95[1]

    if df_preds["pred"].values[0] == 0:
        df_preds = df_preds.iloc[1:]

    for col in ['lower_95','lower_90', 'lower_80', 'lower_50',  'pred',
               'upper_50', 'upper_80', 'upper_90', 'upper_95']: 
        
        df_preds[col] = boxcox.inverse_transform(df_preds[col])[0]
       
    return df_preds

def train_model(df_, geocode, train_start_date, train_end_date):
    '''
    Function to train and save the arima model 
    '''
    df_ = df_.copy()
    
    df_.set_index('dates', inplace = True)

    df_['y'] = df_['y'] + 0.1

    m_arima = Arima(df = df_)

    model = m_arima.train( train_ini_date=train_start_date, train_end_date = train_end_date)

    # Save model
    with open(f'saved_models/arima_{geocode}.pkl', 'wb') as pkl:
        pickle.dump(model, pkl)
    
    # save transf on data
    bc_transformer = m_arima.boxcox
    joblib.dump(bc_transformer, f'saved_models/bc_{geocode}.pkl')

def apply_model(df_, geocode):
    '''
    Function to load and apply the pre trained model 
    '''
    df_ = df_.copy()

    df_.set_index('dates', inplace = True)

    df_['y'] = df_['y'] + 0.1

    bc = joblib.load(f'saved_models/bc_{geocode}.pkl')

    df_.loc[:, "y"] = bc.transform(df_.y)[0]

    with open(f'saved_models/arima_{geocode}.pkl', 'rb') as pkl:
        m_arima = pickle.load(pkl)

    # update the model with the new data:
    m_arima.update(df_)

    start_for = df_.index[-1].strftime("%Y-%m-%d")
    date = get_next_n_weeks(start_for, 3)

    preds_95 = m_arima.predict(3, alpha = 0.05, return_conf_int=True)
    preds_90 = m_arima.predict(3, alpha = 0.1, return_conf_int=True)
    preds_80 = m_arima.predict(3, alpha = 0.2, return_conf_int=True)
    preds_50 = m_arima.predict(3, alpha = 0.5, return_conf_int=True)


    df_for = get_prediction_dataframe(preds_50, preds_80, preds_90, preds_95, date, bc)

    df_for.to_csv(f'forecast_tables/for_arima_{str(Week.fromdate(pd.to_datetime(start_for)))}_{geocode}.csv.gz', index = False)

    return df_for 

