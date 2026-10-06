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

# 3. Differential Convolution: Conv = Smooth(Close) - Smooth(Open)
# Создаем временные DF (или работаем напрямую с колонками — так проще и безопаснее по индексам)
df['Conv_Close'] = gaussian_filter1d(df['close'].values, sigma=1)
df['Conv_Open'] = gaussian_filter1d(df['open'].values, sigma=1)
df['Conv'] = df['Conv_Close'] - df['Conv_Open']

# Дополнительно: производная и Target (для совместимости с прошлой логикой)
df['Conv_Diff'] = df['Conv'].diff()
df['Target_Binary'] = np.where(df['Conv_Diff'] > 0, 1, -1)

# Features (оставляем MACD как в прошлых примерах, но можно адаптировать под Conv)
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]

# Dropna (важно: после gaussian_filter1d пропусков нет, но diff() и shift() создают NaN)
df_clean = df.dropna().copy()

# Train / Test (Last 100)
train_df = df_clean.iloc[:-100]
test_df = df_clean.iloc[-100:].copy()

# 4. Train & Predict
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
model.fit(train_df[features], train_df['Target_Binary'])
test_df['Prediction'] = model.predict(test_df[features])

# --- Торговая логика (Long/Short по Prediction) ---
N = 2
test_df['Position'] = 0
position = 0
hold_counter = 0

for i in range(len(test_df)):
    pred = test_df.iloc[i]['Prediction']
    
    if position == 0:
        if pred == 1:
            position = 1      # Long
            hold_counter = 0
        elif pred == -1:
            position = -1     # Short
            hold_counter = 0
    else:
        hold_counter += 1
        if hold_counter >= N:
            position = 0
            hold_counter = 0
    
    test_df.at[test_df.index[i], 'Position'] = position

# PnL и Equity
test_df['PnL'] = test_df['Position'] * test_df['close'].diff()
test_df['Equity'] = test_df['PnL'].cumsum()

# --- Plot: ТОЛЬКО тестовый участок ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
x_idx = np.arange(len(test_df))

# Ax1: Price и новая Conv (разница сглаженных Close и Open)
ax1.plot(x_idx, test_df['close'], label='Close Price', alpha=0.5)
ax1.plot(x_idx, test_df['Conv'], label='Conv (Smooth Close - Smooth Open)', color='red', linewidth=2)
ax1.set_title('Price & Differential Convolution (Test Set Only)')
ax1.legend(loc='upper left')

# Ax2: Производная Conv
ax2.plot(x_idx, test_df['Conv_Diff'], label='Conv First Derivative', color='green', alpha=0.7)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title('First Derivative of Conv')
ax2.legend(loc='upper left')

# Ax3: Target vs Approximator
ax3.step(x_idx, test_df['Target_Binary'], label='Target (Binary)', where='post', alpha=0.5, color='blue')
ax3.step(x_idx, test_df['Prediction'], label='LGBM Approximator', where='post', linestyle='--', color='orange')
ax3.set_title('Target vs. Approximator')
ax3.legend(loc='upper left')

# Ax4: Equity Curve
ax4.plot(x_idx, test_df['Equity'], label='Equity Curve', color='purple', linewidth=2)
ax4.fill_between(x_idx, test_df['Equity'], alpha=0.3, color='purple')
ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax4.set_title(f'Equity Curve (Hold={N} bars)')
ax4.legend(loc='upper left')

# Подписи оси X (только нижний график)
dates = test_df.index
step = max(1, len(dates) // 15)
tick_positions = np.arange(0, len(dates), step)
tick_labels = [dates[i].strftime('%Y-%m-%d %H:%M') for i in tick_positions]

ax4.set_xticks(tick_positions)
ax4.set_xticklabels(tick_labels, rotation=45, ha='right')

# Убираем подписи на верхних графиках
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])

plt.tight_layout()
plt.show()
mt5.shutdown()
