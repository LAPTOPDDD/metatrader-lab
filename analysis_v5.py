import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Setup
symbol = "EURUSD"
if not mt5.initialize(): quit()

# 1. Fetch
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# 2. Filter weekends AND Reset index to make it CONTINUOUS
df = df[~df.index.dayofweek.isin([5, 6])].copy()
real_times = df.index # Сохраняем реальное время для графиков
df = df.reset_index(drop=True) # Теперь индекс 0, 1, 2...

# 3. Indicators on continuous, clean data
df['Conv'] = gaussian_filter1d(df['close'].values, sigma=12)
df['Conv_Diff'] = df['Conv'].diff()
df['Target_Binary'] = np.where(df['Conv_Diff'] > 0, 1, -1)

# Features
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]
df.dropna(inplace=True)

# Отрезаем реальное время по тому же индексу, что и данные после dropna
real_times = real_times[df.index]

# Train / Test (Last 100)
train_df = df.iloc[:-100]
test_df = df.iloc[-100:]
train_times = real_times[:-100]
test_times = real_times[-100:]

# 4. Train
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
model.fit(train_df[features], train_df['Target_Binary'])

# Predict
test_df['Prediction'] = model.predict(test_df[features])

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Use real_times instead of df index
ax1.plot(real_times, df['close'], label='Close Price', alpha=0.5)
ax1.plot(real_times, df['Conv'], label='Gaussian Conv', color='red')
ax1.set_title('Close Price & Gaussian Convolution (Pure Continuous)')
ax1.legend()

ax2.plot(real_times, df['Conv_Diff'], label='Conv First Derivative', color='green', alpha=0.7)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title('First Derivative')
ax2.legend()

ax3.step(test_times, test_df['Target_Binary'], label='Target (Binary)', where='post', alpha=0.5)
ax3.step(test_times, test_df['Prediction'], label='LGBM Approximator', where='post', linestyle='--')
ax3.set_title('Target vs. Approximator')
ax3.legend()

plt.tight_layout()
plt.show()
mt5.shutdown()
