import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Setup
symbol = "USDJPY"
if not mt5.initialize():
    quit()

# 1. Fetch
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 400)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# 2. Filter weekends (Sat=5, Sun=6) — ОСТАВЛЯЕМ ВСЁ, КРОМЕ ВЫХОДНЫХ
df = df[~df.index.dayofweek.isin([5, 6])].copy()

# 3. Indicators — везде 'close' (строчная)
df['Conv'] = gaussian_filter1d(df['close'].values, sigma=2)
df['Conv_Diff'] = df['Conv'].diff()
df['Target_Binary'] = np.where(df['Conv_Diff'] > 0, 1, -1)

# Features
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]

# Dropna
df_clean = df.dropna().copy()

# Train / Test (Last 150)
train_df = df_clean.iloc[:-150]
test_df = df_clean.iloc[-150:].copy()

# Проверка: если данных мало — выводим предупреждение
if len(train_df) < 50 or len(test_df) < 50:
    print("ВНИМАНИЕ: слишком мало данных после фильтрации и dropna!")

# 4. Train & Predict
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
model.fit(train_df[features], train_df['Target_Binary'])
test_df['Prediction'] = model.predict(test_df[features])

# --- Торговая логика: вход при Prediction == 1, удержание N баров ---
N = 1  # по умолчанию

test_df['Signal'] = (test_df['Prediction'] == 1).astype(int)
test_df['Position'] = 0

position = 0
hold_counter = 0

for i in range(len(test_df)):
    if position == 0:
        if test_df.iloc[i]['Signal'] == 1:
            position = 1
            hold_counter = 0
    else:
        hold_counter += 1
        # Закрываем, когда продержали БОЛЬШЕ N баров
        if hold_counter > N:
            position = 0
            hold_counter = 0
    
    test_df.at[test_df.index[i], 'Position'] = position

# PnL и Equity
test_df['PnL'] = test_df['Position'] * test_df['close'].diff()
test_df['Equity'] = test_df['PnL'].cumsum()

# --- Plot: ТОЛЬКО тестовый участок ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
x_idx = np.arange(len(test_df))

# Ax1: Price & Conv
ax1.plot(x_idx, test_df['close'], label='Close Price', alpha=0.5)
ax1.plot(x_idx, test_df['Conv'], label='Gaussian Conv', color='red')
ax1.set_title('Close Price & Gaussian Convolution (Test Set Only)')
ax1.legend(loc='upper left')

# Ax2: Conv Diff
ax2.plot(x_idx, test_df['Conv_Diff'], label='Conv First Derivative', color='green', alpha=0.7)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title('First Derivative (Test Set Only)')
ax2.legend(loc='upper left')

# Ax3: Target vs Approximator
ax3.step(x_idx, test_df['Target_Binary'], label='Target (Binary)', where='post', alpha=0.5, color='blue')
ax3.step(x_idx, test_df['Prediction'], label='LGBM Approximator', where='post', linestyle='--', color='orange')
ax3.set_title('Target vs. Approximator (Test Set Only)')
ax3.legend(loc='upper left')

# Ax4: Equity Curve
ax4.plot(x_idx, test_df['Equity'], label='Equity Curve', color='purple', linewidth=2)
ax4.fill_between(x_idx, test_df['Equity'], alpha=0.3, color='purple')
ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax4.set_title(f'Equity Curve (Test Set Only, Hold={N} bars)')
ax4.legend(loc='upper left')

# Настройка подписей оси X
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
