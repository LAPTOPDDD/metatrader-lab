import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

# --- Конфигурация ---
N_BARS_HOLD = 20 # Увеличим время удержания, так как у нас теперь есть TP
ATR_MULTIPLIER = 1.0 # База для стопа
RR_RATIO = 4.0 # <--- ТВОЯ ПЕРЕМЕННАЯ: попробуй 3, 4, 5
symbol = "EURUSD"

# Initialize MT5
if not mt5.initialize(): quit()

# Data
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 5000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Feature Engineering
period_n = 20
df['Rel_Price'] = (df['close'] - df['low'].rolling(period_n).min()) / \
                  (df['high'].rolling(period_n).max() - df['low'].rolling(period_n).min())
df['SMA_50'] = df['close'].rolling(50).mean()
df['Dist_SMA50'] = (df['close'] - df['SMA_50']) / df['SMA_50']
df['Momentum'] = df['close'].diff(5)
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['MACD_dir'] = np.where(df['MACD'].diff() >= 0, 1, -1)

features = ['Rel_Price', 'Dist_SMA50', 'Momentum', 'MACD_dir', 'ATR']
for i in range(1, 6):
    df[f'Lag_{i}'] = df['MACD_dir'].shift(i)
    features.append(f'Lag_{i}')

df['Gaussian_conv'] = gaussian_filter1d(df['close'], sigma=2)
df['Target_class'] = np.where(df['Gaussian_conv'].diff() > 0, 1, -1)
df.dropna(inplace=True)

# Train/Test
train_df = df.iloc[:-150]
test_df = df.iloc[-150:].copy()

model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.01, num_leaves=31, max_depth=6, verbosity=-1, random_state=42)
model.fit(train_df[features], train_df['Target_class'])

# Prediction
probs = model.predict_proba(test_df[features])
test_df['Signal'] = np.where(probs[:, 1] > 0.65, 1, np.where(probs[:, 0] > 0.65, -1, 0))

# --- BACKTESTING WITH SL/TP ---
test_df['Entry_Signal'] = test_df['Signal'].shift(1)
equity_returns = []
active_pos = 0

for i in range(len(test_df)):
    price = test_df['close'].iloc[i]
    atr = test_df['ATR'].iloc[i]
    signal = test_df['Entry_Signal'].iloc[i]
    
    if active_pos != 0:
        pos_counter -= 1
        # Логика: выход если SL, TP или время вышло
        if active_pos == 1:
            if price <= stop_price or price >= tp_price or pos_counter == 0:
                equity_returns.append(active_pos * (price - entry_price))
                active_pos = 0
            else: equity_returns.append(0)
        elif active_pos == -1:
            if price >= stop_price or price <= tp_price or pos_counter == 0:
                equity_returns.append(active_pos * (price - entry_price))
                active_pos = 0
            else: equity_returns.append(0)
    elif signal != 0:
        active_pos = signal
        pos_counter = N_BARS_HOLD
        entry_price = price
        sl_dist = ATR_MULTIPLIER * atr
        stop_price = price - (signal * sl_dist)
        tp_price = price + (signal * sl_dist * RR_RATIO)
        equity_returns.append(0)
    else: equity_returns.append(0)

test_df['Equity_Return'] = equity_returns
test_df = test_df[~test_df.index.dayofweek.isin([5, 6])]
test_df['Equity'] = test_df['Equity_Return'].cumsum()

plt.figure(figsize=(14, 6))
plt.plot(test_df.index, test_df['Equity'], label=f'Equity (RR={RR_RATIO})', color='purple', linewidth=2)
plt.title(f'Strategy with SL and TP (RR={RR_RATIO}): {symbol}')
plt.grid(True)
plt.legend()
plt.show()
mt5.shutdown()
