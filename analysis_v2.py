import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Setup
symbol = "EURUSD"
if not mt5.initialize(): quit()

# Fetch enough data
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 2500)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# 1. Filter out weekends
df = df[~df.index.dayofweek.isin([5, 6])]

# 2. Indicator Calculation
df['Gaussian_conv'] = gaussian_filter1d(df['close'], sigma=2)
df['Target_Binary'] = np.where(df['Gaussian_conv'].diff() > 0, 1, -1)

# Features (MACD, RSI)
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]

# Drop NaNs
df.dropna(inplace=True)

# Train / Test (Last 100)
train_df = df.iloc[:-100]
test_df = df.iloc[-100:].copy()

# 3. Train Approximator
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
model.fit(train_df[features], train_df['Target_class'] if 'Target_class' in train_df.columns else train_df['Target_Binary'])

# Predict
test_df['Prediction'] = model.predict(test_df[features])

# Plot (Last 200 bars)
plot_df = df.tail(200)
test_plot_df = test_df.tail(100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot 1: Close prices and Gaussian convolution
ax1.plot(plot_df.index, plot_df['close'], label='Close Price', alpha=0.6)
ax1.plot(plot_df.index, plot_df['Gaussian_conv'], label='Gaussian Conv', color='red')
ax1.set_title('EURUSD H1: Close Price & Gaussian Convolution (Weekends Filtered)')
ax1.legend()

# Plot 2: Binary Target vs Approximator
ax2.step(test_plot_df.index, test_plot_df['Target_Binary'], label='Target (Binary Conv Change)', where='post', alpha=0.5)
ax2.step(test_plot_df.index, test_plot_df['Prediction'], label='LGBM Approximator', where='post', linestyle='--')
ax2.set_title('Target vs. LightGBM Approximator (Test Area - Last 100 Bars)')
ax2.legend()

plt.tight_layout()
plt.show()

mt5.shutdown()
