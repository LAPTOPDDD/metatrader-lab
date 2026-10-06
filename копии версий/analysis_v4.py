import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Setup
symbol = "EURUSD"
if not mt5.initialize(): quit()

# 1. Fetch data
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# 2. Filter weekends IMMEDIATELY
df = df[~df.index.dayofweek.isin([5, 6])]

# 3. Indicators on continuous data
# Gaussian convolution
df['Conv'] = gaussian_filter1d(df['close'], sigma=2)

# First Derivative of Convolution
df['Conv_Diff'] = df['Conv'].diff()

# Binary form of the Derivative
df['Target_Binary'] = np.where(df['Conv_Diff'] > 0, 1, -1)

# Features (MACD Lags)
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]

# Drop NaNs after diff/shift/rolling
df.dropna(inplace=True)

# Train / Test (Last 100 bars)
train_df = df.iloc[:-100]
test_df = df.iloc[-100:].copy()

# 3. Train Approximator
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
model.fit(train_df[features], train_df['Target_Binary'])

# Predict
test_df['Prediction'] = model.predict(test_df[features])

# Plot (Last 200 bars)
plot_df = df.tail(200)
test_plot_df = test_df.tail(100)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Plot 1: Close + Conv
ax1.plot(plot_df.index, plot_df['close'], label='Close Price', alpha=0.5)
ax1.plot(plot_df.index, plot_df['Conv'], label='Gaussian Conv', color='red')
ax1.set_title('Close Price & Gaussian Convolution')
ax1.legend()

# Plot 2: First Derivative (Continuous)
ax2.plot(test_plot_df.index, test_plot_df['Conv_Diff'], label='Conv First Derivative', color='green', alpha=0.7)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title('First Derivative of Convolution')
ax2.legend()

# Plot 3: Binary Target vs Approximator
ax3.step(test_plot_df.index, test_plot_df['Target_Binary'], label='Target (Binary Derivative)', where='post', alpha=0.5)
ax3.step(test_plot_df.index, test_plot_df['Prediction'], label='LGBM Approximator', where='post', linestyle='--')
ax3.set_title('Target vs. LightGBM Approximator (Last 100 Bars)')
ax3.legend()

plt.tight_layout()
plt.show()

mt5.shutdown()
