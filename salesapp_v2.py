# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:59:24 2022

@author: lemos3
"""

import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import datetime
from datetime import date
import streamlit as st
# import os

#DEFINE FUNCTION TO FIND FILE
# def find(name, path):
#     for root, dirs, files in os.walk(path):
#         if name in files:
#             return True

#  DEFINE TODAY'S DATE
today = date.today()
d = datetime.timedelta(days = 400.0)
from_day = today-d


# IMPORT STORE TABLE
stores = pd.read_csv('store_master.csv',dtype='str', usecols=['CLUSTER', 'COUNTRY', 'TABLEAU STORE NAME', 'ERP STORE CODE MAPPING',
        'BUSINESS MODEL', 'CHANNEL', 'FRANCHISE PARTNER NAME',
        'DISTRICT'])

stores.drop_duplicates(subset=['ERP STORE CODE MAPPING'], inplace=True)
stores.dropna(subset=['ERP STORE CODE MAPPING'], inplace=True)
stores.dropna(subset=['CLUSTER'], inplace=True)
stores.rename(columns={'ERP STORE CODE MAPPING':"Store Code"},inplace=True)
stores.set_index('Store Code')


# FIND & ADD LAST WEEK'S DATA - DO NOT USE + NEEDS TO BE PLACED LATER IN THE CODE

# df = df.merge(stores,on='Store Code', how='left')
# if find('lastweek.txt','/'):
    # lw = pd.read_csv('lastweek.txt',sep='\t')
    # df = df.append(lw)
    # os.remove('lastweek.txt')
    # df.to_csv('data.txt',index=False, sep='\t')


#  IMPORT DATA
df = pd.read_csv('data.txt',sep='\t', usecols=['Store Code','Date','Attribute','Value'])
df = df[df['Attribute']=='Local - Without Tax']
df['Store Code'] = df['Store Code'].astype('str')
# df = df[df['trading_status']=='OPEN']
df.dropna(subset='Store Code', inplace=True)
df.reset_index(drop=True, inplace=True)
df.sort_values(by="Store Code",inplace=True)
# MERGING TWO DATAFRAMES TO HAVE ALL THE STORE INFORMATION FOR FILTERING

master = pd.merge(df,stores,how='left',validate='m:1')
master.dropna(subset=['COUNTRY'],inplace=True)
master.reset_index(drop=True, inplace=True)

# CREATING LISTS & DICTS FOR SELECTORS/FILTER
base_list = ['All']
nest = stores[['CLUSTER', 'COUNTRY','DISTRICT','TABLEAU STORE NAME','BUSINESS MODEL','CHANNEL','Store Code']]

levels = nest.groupby('CLUSTER').apply(lambda a: dict(a.groupby('COUNTRY').apply(lambda x: dict(x.groupby('DISTRICT').apply(lambda y: dict(y.groupby('TABLEAU STORE NAME')['Store Code'].count()))))))
levels = levels.to_dict()

levels['All']={}
levels['All']['All']={}
levels['All']['All']['All'] = 'All'

cluster_list = list(stores.CLUSTER.dropna().unique())
# country_list = list(stores.COUNTRY.dropna().unique())
# district_list = list(stores.DISTRICT.dropna().unique())


cluster_list.sort()
# country_list.sort()
# district_list.sort()

cluster_list = base_list + cluster_list
# country_list = base_list + country_list
# district_list = base_list + district_list

# APP CODE

st.title('Sales Forecast App')

with st.sidebar:
    st.subheader('Make your selections')
    cluster = st.selectbox('Select a cluster from the dropdown list',(cluster_list))
    country = st.selectbox('Select a country from the dropdown list',(list(levels[cluster])))
    if country =='All':
        district = st.selectbox('Select a district from the dropdown list',(base_list))
    else: district = st.selectbox('Select a district from the dropdown list',(base_list+list(levels[cluster][country])))
    
#     btn_1 = st.button('Enter')
# if btn_1:
    
if cluster != 'All':
    master = master[master['CLUSTER']==cluster]
if country != 'All':
    master = master[master['COUNTRY']==country]
if district != 'All':
    master = master[master['DISTRICT']==district]
# else:
#     df = df
# st.write(df.describe())
#st.write(df)
# if country != 'All':
#     master = master[master['COUNTRY']==country]
# else:
#     df = df



sales = master[['Date','Value']]
sales['Date']=pd.to_datetime(sales['Date'])

# traffic = df[['Date','FF']]
# traffic['Date']=pd.to_datetime(sales['Date'])
# trans = df[['Date','Transactions']]
# trans['Date']=pd.to_datetime(sales['Date'])
# units = df[['Date','Units Sold']]
# units['Date']=pd.to_datetime(sales['Date'])


sales = sales.set_index(['Date'])
sales = sales.loc['2020-11-30': , : ]
# traffic = traffic.set_index(['Date'])
# traffic = traffic.loc['2019-11-29': , : ]

# trans = trans.set_index(['Date'])
# trans = trans.loc['2019-11-29': , : ]

# units = units.set_index(['Date'])
# units = units.loc['2019-11-29': , : ]



sales = sales.resample('D').sum()
daily_sales = sales.copy()
sales['Rolling'] = sales["Value"].rolling(window=30).mean()
sales['Expanded'] = sales["Value"].expanding().mean()

# daily_ff = traffic.resample('D').sum()
# daily_mean_ff = daily_ff.copy()
# daily_mean_ff['Rolling'] = daily_mean_ff["FF"].rolling(window=30).mean()
# daily_mean_ff['Expanded'] = daily_mean_ff["FF"].expanding().mean()

# daily_trans = trans.resample('D').sum()
# daily_mean_trans = daily_trans.copy()
# daily_mean_trans['Rolling'] = daily_mean_trans["Transactions"].rolling(window=30).mean()
# daily_mean_trans['Expanded'] = daily_mean_trans["Transactions"].expanding().mean()

# daily_units = units.resample('D').sum()
# daily_mean_units = daily_units.copy()
# daily_mean_units['Rolling'] = daily_mean_units["Units Sold"].rolling(window=30).mean()
# daily_mean_units['Expanded'] = daily_mean_units["Units Sold"].expanding().mean()


# if cluster != 'All':
#     st.subheader(f'Sales in {cluster} in the last 13 Months')
# elif country != 'All':
#     st.subheader(f'Sales in {country} in the last 13 Months')
# elif district != 'All':
#     st.subheader(f'Sales in {district} in the last 13 Months')
# else:
#     st.subheader('Sales in the last 13 Months')
st.subheader('Sales trends, starting FY2021')

st.line_chart(sales)
# st.line_chart(daily_mean_ff)
# st.line_chart(daily_mean_trans)
# st.line_chart(daily_mean_units)


renaming_dict = {"Date" : "ds", "Value" : "y"}
# train_df = daily_sales.loc['2021': ,:]
# df_prophet = sales['Value']
daily_sales = daily_sales.reset_index().rename(renaming_dict, axis=1)

country_codes = pd.read_csv('iso_codes.csv')

country_key = list(country_codes['Country'])
country_value = list(country_codes['Alpha-2 code'])

dict_country = {country_key[i]: country_value[i] for i in range(len(country_key))}


if country != 'All':
    country_name = dict_country[country]

# col2,col3 = st.columns(1,2)
# with col2:
days = st.selectbox('Select number of days for prediction', (30,60,90,120,150))
# with col3:    

use_holidays = st.radio('Use Country Holidays?', ('Yes','No'))

btn = st.button('Predict')

if btn:
    from prophet import Prophet

    model = Prophet()    
    if use_holidays=='Yes':
        model.add_country_holidays(country_name=country_name)
    model.fit(daily_sales)   
    future = model.make_future_dataframe(periods=days, freq='D')

    forecast = model.predict(future)

    st.subheader(f'Predictions for the next {days} days.')

    st.write(model.plot(forecast))
    
    
    with st.expander('Detailed Figures'):
        st.table(forecast[[	'ds','trend','yhat_lower','yhat_upper','trend_lower','trend_upper']].tail(days))