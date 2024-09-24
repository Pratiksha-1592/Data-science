#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[8]:


df = pd.read_csv(r"C:\Users\Pratiksha\OneDrive\Documents\DATA SCIENCE\time_series_analysis/BAJFINANCE.csv")


# In[10]:


df.head()


# In[38]:


df.dtypes


# In[16]:


df.set_index('Date',inplace=True),


# In[18]:


df['VWAP'].plot()


# In[22]:


df['Volume'].plot()


# In[24]:


df.shape


# In[26]:


df.isna().sum()


# In[28]:


df.dropna(inplace=True)


# In[30]:


df.isna().sum()


# In[32]:


df.shape


# In[34]:


data=df.copy()


# In[36]:


data.dtypes


# In[40]:


data.columns


# In[42]:


lag_features=['High','Low','Volume','Turnover','Trades']
window1=3
window2=7


# In[44]:


for feature in lag_features:
    data[feature+'rolling_mean_3']=data[feature].rolling(window=window1).mean()
    data[feature+'rolling_mean_7']=data[feature].rolling(window=window2).mean()


# In[46]:


for feature in lag_features:
    data[feature+'rolling_std_3']=data[feature].rolling(window=window1).std()
    data[feature+'rolling_std_7']=data[feature].rolling(window=window2).std()


# In[48]:


data.head()


# In[50]:


data.columns


# In[54]:


data.isna().sum()


# In[56]:


data.dropna(inplace=True)


# In[58]:


data.isna().sum()


# In[110]:


ind_features=['Highrolling_mean_3', 'Highrolling_mean_7',
       'Lowrolling_mean_3', 'Lowrolling_mean_7', 'Volumerolling_mean_3',
       'Volumerolling_mean_7', 'Turnoverrolling_mean_3',
       'Turnoverrolling_mean_7', 'Tradesrolling_mean_3',
       'Tradesrolling_mean_7', 'Highrolling_std_3', 'Highrolling_std_7',
       'Lowrolling_std_3', 'Lowrolling_std_7', 'Volumerolling_std_3',
       'Volumerolling_std_7', 'Turnoverrolling_std_3', 'Turnoverrolling_std_7',
       'Tradesrolling_std_3', 'Tradesrolling_std_7']


# In[78]:


training_data=data[0:1800]
test_data=data[1800:]


# In[80]:


training_data


# In[112]:


get_ipython().system('pip install pmdarima')


# In[171]:


import pmdarima
print(pmdarima.__version__)


# In[173]:


import sys
print(sys.version)


# In[175]:


from pmdarima import auto_arima


# In[177]:


model = auto_arima(y = training_data['VWAP'] , X = training_data[ind_features], trace=True)


# In[181]:


forecast = model.predict(n_periods=len(test_data), X = test_data[ind_features])


# In[183]:


test_data['Forecast_ARIMA'] = forecast.values


# In[185]:


test_data[['VWAP' , 'Forecast_ARIMA']].plot(figsize=(14,7))


# In[187]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
print(np.sqrt(mean_squared_error(test_data['VWAP'],test_data['Forecast_ARIMA'])))


# In[189]:


print(mean_absolute_error(test_data['VWAP'],test_data['Forecast_ARIMA']))


# In[ ]:




