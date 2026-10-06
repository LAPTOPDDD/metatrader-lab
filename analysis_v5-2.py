import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Setup
symbol = "EURUSD"
if not mt5.initialize():
    quit()

# 1. Fetch
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# 2. Filter weekends (Sat=5, Sun=6)
df = df[~df.index.dayofweek.isin([5, 6])].copy()

# 3. Indicators
df['Conv'] = gaussian_filter1d(df['close'].values, sigma=12)
df['Conv_Diff'] = df['Conv'].diff()
df['Target_Binary'] = np.where(df['Conv_Diff'] > 0, 1, -1)

# Features
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]

# Dropna
df_clean = df.dropna().copy()

# Train / Test (Last 100)
train_df = df_clean.iloc[:-100]
test_df = df_clean.iloc[-100:]

# 4. Train
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
model.fit(train_df[features], train_df['Target_Binary'])

# Predict
test_df['Prediction'] = model.predict(test_df[features])

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# --- Ось X: порядковый индекс (0,1,2,...) вместо дат ---
x_idx_all = np.arange(len(df_clean))
x_idx_test = np.arange(len(df_clean) - 100, len(df_clean))  # индексы последних 100 баров

# Ax1: Price & Conv (все данные)
ax1.plot(x_idx_all, df_clean['close'], label='Close Price', alpha=0.5)
ax1.plot(x_idx_all, df_clean['Conv'], label='Gaussian Conv', color='red')
ax1.set_title('Close Price & Gaussian Convolution (Weekdays Only)')
ax1.legend(loc='upper left')

# Ax2: Conv Diff (все данные)
ax2.plot(x_idx_all, df_clean['Conv_Diff'], label='Conv First Derivative', color='green', alpha=0.7)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title('First Derivative')
ax2.legend(loc='upper left')

# Ax3: Target & Prediction (только последние 100 баров)
ax3.step(x_idx_test, test_df['Target_Binary'], label='Target (Binary)', where='post', alpha=0.5, color='blue')
ax3.step(x_idx_test, test_df['Prediction'], label='LGBM Approximator', where='post', linestyle='--', color='orange')
ax3.set_title('Target vs. Approximator (Last 100 Bars Only)')
ax3.legend(loc='upper left')

# --- Настройка подписей оси X (только некоторые даты) ---
dates = df_clean.index
step = max(1, len(dates) // 20)  # ~20 меток на весь график
tick_positions = np.arange(0, len(dates), step)
tick_labels = [dates[i].strftime('%Y-%m-%d %H:%M') for i in tick_positions]

ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=45, ha='right')

# Подписи на верхних графиках не нужны, т.к. ось общая
ax1.set_xticklabels([])
ax2.set_xticklabels([])

plt.tight_layout()
plt.show()
mt5.shutdown()
