# DMV Practical 7: Stock Market Time Series Analysis & Forecasting
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load Dataset
df = pd.read_csv("Google_Stock_Train (2010-2022).csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# 2. Moving Averages
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# 3. Plot Historical Trend & Moving Averages
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label='Close Price', color='black', alpha=0.6)
plt.plot(df.index, df['MA_20'], label='20-Day SMA', color='blue')
plt.plot(df.index, df['MA_50'], label='50-Day SMA', color='orange')
plt.plot(df.index, df['EMA_20'], label='20-Day EMA', color='red', linestyle='--')
plt.title("Stock Price with Moving Averages")
plt.legend()
plt.show()

# 4. Train-Test Split & Forecast
split = int(len(df) * 0.8)
train = df['Close'][:split]
test = df['Close'][split:]

try:
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train, order=(1, 1, 1)).fit()
    forecast = model.forecast(steps=len(test))
except ImportError:
    from sklearn.linear_model import Ridge
    t_tr = np.arange(len(train)).reshape(-1, 1)
    t_ts = np.arange(len(train), len(train)+len(test)).reshape(-1, 1)
    forecast = pd.Series(Ridge().fit(t_tr, train.values).predict(t_ts), index=test.index)

print(f"RMSE: {np.sqrt(mean_squared_error(test, forecast)):.2f}")
print(f"MAE:  {mean_absolute_error(test, forecast):.2f}")

plt.figure(figsize=(9, 4))
plt.plot(test.index, test, label='Actual Test')
plt.plot(test.index, forecast, label='Forecast', color='green', linestyle='--')
plt.title("Stock Price Forecast vs Actual")
plt.legend()
plt.show()
