# Fuzzy-Time-Series-Forecasting-


# 1. Load and clean data

df = pd.read_csv('/content/NIFTY 50_daily_data.csv')

# 2. Convert dates (ONCE is enough)
# Correct format including time


df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')  # For "DD-MM-YYYY HH:MM" format
# Filter for the last 10 years

end_date = df['date'].max()
start_date = end_date - pd.DateOffset(years=10)
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

## Logic:

First we load the raw CSV data into a pandas DataFrame

Convert the date string into proper datetime objects for time series operations

Find the most recent date in the data (end_date)

Calculate start_date as 10 years prior to end_date

Filter the DataFrame to only include this 10-year window

## Why?

Working with datetime objects enables proper time-based operations

Limiting to 10 years gives us enough history while keeping data relevant

Removes very old data that might have different market characteristics


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

## Logic:

Find the minimum and maximum closing prices in our data

Calculate a 10% margin buffer on both sides

Define our universe [u_min, u_max] as the range plus buffers

## Why?

The universe must contain all possible values plus room for future extremes

10% margin provides space for forecasts outside historical range

Ensures we can handle values slightly beyond what we've seen


# partitioning the universe into intervals

num_interval=15
interval_width=(u_max-u_min)/num_interval

interval=[]

for i in range(num_interval):
  lower=u_min+i*interval_width
  upper=lower+interval_width
  interval.append((lower,upper))

interval

## Logic:

Choose to create 15 intervals (can be adjusted)

Calculate width of each interval as total range divided by number of intervals

Create non-overlapping intervals that cover the entire universe

## Why?

15 intervals provides reasonable granularity without being too sparse or dense

Equal-width intervals simplify calculations and interpretation

Complete coverage ensures any value can be classified


# creating fuzzy sets


fuzzy_set={}

for  idx,(lower,upper) in enumerate(interval,1):
  set_name=f'F{idx}'
  fuzzy_set[set_name]=(lower,upper)

fuzzy_set.items()


## Logic:

Create a dictionary to store our fuzzy sets

For each interval, create a named fuzzy set (F1, F2,...F15)

Store each set's lower and upper bounds

## Why?

Dictionary provides easy lookup of interval bounds by name

Numbered sets (F1-F15) allow ordered reference to ranges

Stores the mapping between linguistic terms and numerical ranges


# fuzzyfying historical data

def assign_fuzzy(value):
  for name,(lower,upper) in fuzzy_set.items():
    if lower<=value<upper:
      return name
  return None

df['fuzzy']=df['close'].apply(assign_fuzzy)

df.head()

## Logic:

Define function to classify a numerical value into a fuzzy set

Check which interval contains the value

Return the corresponding fuzzy set name

Apply this to all closing prices to create new 'fuzzy' column

## Why?

Converts numerical data to linguistic terms (F1, F2, etc.)

Enables pattern recognition at abstract level rather than precise numbers

Creates the foundation for fuzzy relationship analysis


# establishing fuzzy logical relationships

flrs=[]

for i in range(1,len(df)):
  prev=df.iloc[i-1]['fuzzy']
  curr=df.iloc[i]['fuzzy']
  if prev and curr:
    flrs.append((prev,curr))

## Logic:

Initialize empty list for Fuzzy Logical Relationships (FLRs)

Iterate through the data to find consecutive day pairs

For each pair, record the relationship (previous day → current day)

Only store valid relationships (non-null values)

## Why?

Captures how the system transitions between states

Each tuple represents a "if F1 today then F2 tomorrow" pattern

Forms the basis for predicting future states



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

## Logic:

Create dictionary to store Fuzzy Logical Relationship Groups (FLRGs)

For each relationship, check if we've seen this "previous" state before

If yes, add the "current" state to its list of outcomes (if not already present)

If no, create new entry with this "current" state

## Why?

Groups all possible outcomes for each starting state

Shows what typically follows each fuzzy set

For forecasting, we'll consider all historical outcomes
  

# mid_point function

def get_midpoint(fuzzy_label):
  lower,upper=fuzzy_set[fuzzy_label]
  return (lower+upper)/2



## Logic:

Takes a fuzzy set name (like 'F3')

Looks up its lower and upper bounds

Returns the exact midpoint of the range

## Why?

Needed to convert fuzzy forecasts back to numerical values

Uses center of interval as representative value

Simple method for defuzzification

  

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




## Logic:

Initialize empty forecasts list

For each day (after the first):
a. Get previous day's fuzzy set
b. If we have relationships for this set:

Get all possible next states

Convert each to its midpoint

Forecast is average of these midpoints
c. If no relationships (new state):

Use midpoint of current state as forecast

Store all forecasts

## Why?

Average of historical outcomes provides balanced prediction

Handles cases where multiple transitions are possible

Falls back to midpoint when no history available (smoothing)



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
# forecasting accuracy testing 

df_evaluate = df1.dropna(subset=['forecast'])
y_true = df_evaluate['close']
y_pred = df_evaluate['forecast']
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)


## Logic:

Create clean copy of forecast data

Remove any rows with missing values

Extract true values and predictions

Calculate:

MAE: Average absolute difference

MSE: Average squared difference

RMSE: Root of MSE (same units as original data)

## Why?

Quantifies forecast accuracy

Different metrics emphasize different aspects of error

Allows comparison with other forecasting methods


