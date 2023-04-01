import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

st.title('Forecasting')

st.write('Import the time series csv file here. Columns must be labeled ds and y. The input to Prophet is always a dataframe with two columns: ds and y. ', type='csv')
st.write('CSV file can be updated and reuploaded any number of times inorder to get prediction and forecasting depends on only two factors date(ds) and target column(y)')

l = ['Rice','sugar','oil']
use_defo = st.selectbox('Select Dataset',l)
if use_defo == "":
    st.write("Choose one")
if use_defo == 'Rice':
    df = 'Riceprice.csv'
elif use_defo == 'sugar':
    df = 'sugar.csv'
else :
    df = 'oil.csv'


if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    
    st.write(data)
    
    max_date = data['ds'].max()
    #st.write(max_date)



periods_input = st.number_input('How many periods would you like to forecast into the future?',
min_value = 1, max_value = 31)

if df is not None:
    m = Prophet()
    m.fit(data)

"""
### Step 3: Visualize Forecast Data

The below visual shows future predicted values. "yhat" is the predicted value.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
 
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    st.write('Trend is like similarity which is observed from given data and plot depends on datestamp') 
