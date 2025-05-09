# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18wmN8WUhCJz1xplnRfXJuwO3iTMlY5BK
"""

# nifty 50 (2015-2025) fuzzy time series forecasting

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and clean data
df = pd.read_csv('/content/NIFTY 50_daily_data.csv')

# 2. Convert dates (ONCE is enough)
# Correct format including time
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')  # For "DD-MM-YYYY HH:MM" format
# Filter for the last 10 years
end_date = df['date'].max()
start_date = end_date - pd.DateOffset(years=10)
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]



# Plot the closing prices
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['close'], label='NIFTY 50 Close Price')
plt.title('NIFTY 50 Closing Prices - Last 10 Years')
plt.xlabel('date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

# defining the universe of  discourse

data_min=df['close'].min()
data_max=df['close'].max()
margin=0.1*(data_max-data_min)

u_max=data_max+margin
u_min=data_min-margin


u=[u_min,u_max]

# partitioning the universe into intervals

num_interval=15
interval_width=(u_max-u_min)/num_interval

interval=[]

for i in range(num_interval):
  lower=u_min+i*interval_width
  upper=lower+interval_width
  interval.append((lower,upper))

interval

# creating fuzzy sets


fuzzy_set={}

for  idx,(lower,upper) in enumerate(interval,1):
  set_name=f'F{idx}'
  fuzzy_set[set_name]=(lower,upper)

fuzzy_set.items()

# fuzzyfying historical data

def assign_fuzzy(value):
  for name,(lower,upper) in fuzzy_set.items():
    if lower<=value<upper:
      return name
  return None

df['fuzzy']=df['close'].apply(assign_fuzzy)

df.head()

# establishing fuzzy logical relationships

flrs=[]

for i in range(1,len(df)):
  prev=df.iloc[i-1]['fuzzy']
  curr=df.iloc[i]['fuzzy']
  if prev and curr:
    flrs.append((prev,curr))

# creating flrgs

flrgs={}

for prev,curr in flrs:
  if prev in flrgs:
    if curr not in flrgs[prev]:
      flrgs[prev].append(curr)
  else:
    flrgs[prev]=[curr]


print('fuzzy logical relationship groups ')


for prev , curr in flrgs.items():
  print(f"{prev} -> {','.join(curr)}")

# mid_point function

def get_midpoint(fuzzy_label):
  lower,upper=fuzzy_set[fuzzy_label]
  return (lower+upper)/2

# forecasting

forecasts=[]


for i in range(1,len(df)):
  prev_fuzzy=df.iloc[i-1]['fuzzy']
  if prev_fuzzy in flrgs:
    curr=flrgs[prev_fuzzy]
    midpoint=[get_midpoint(c) for c in curr]
    forecast=np.mean(midpoint)

  else:
    forecast=get_midpoint(prev_fuzzy)
  forecasts.append(forecast)

df

# allign forecast with dates

#df=df.reset_index()


forecast_dates=df['date'][1:].reset_index(drop=True)
forecast_df=pd.DataFrame({'date':forecast_dates,'forecast':forecasts})


# merge with original data

df_forecast=pd.merge(df,forecast_df,on='date',how='left')

df_forecast

# displaying results

#print(df_forecast[['date','Close','forecast']].head(10))


plt.plot(df_forecast['forecast'])

plt.plot(df_forecast['date'], df_forecast['close'])
plt.plot(df_forecast['date'], df_forecast['forecast'])

plt.legend()

plt.show()



df1 = df_forecast.copy()

df1.dropna(inplace=True)

from sklearn.metrics import mean_absolute_error, mean_squared_error


df_evaluate = df1.dropna(subset=['forecast'])
y_true = df_evaluate['close']
y_pred = df_evaluate['forecast']
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)


#mae - 866.72 -> -483.31
#mse - 995412.441 -> -785918.38
rmse - 457