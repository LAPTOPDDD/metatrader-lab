import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

# --- Конфигурация ---
N_BARS_HOLD = 3  # Переменная для подбора
symbol = "EURUSD"
timestamp = int(time.time())

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
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['MACD_dir'] = np.where(df['MACD'].diff() >= 0, 1, -1)

features = ['Rel_Price', 'Dist_SMA50', 'Momentum', 'MACD_dir']
for i in range(1, 6):
    df[f'Lag_{i}'] = df['MACD_dir'].shift(i)
    features.append(f'Lag_{i}')

df['Gaussian_conv'] = gaussian_filter1d(df['close'], sigma=2)
df['Target_class'] = np.where(df['Gaussian_conv'].diff() > 0, 1, -1)
df.dropna(inplace=True)

# Train/Test
train_df = df.iloc[:-350]
test_df = df.iloc[-350:].copy()

model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.01, num_leaves=31, max_depth=6, verbosity=-1, random_state=42)
model.fit(train_df[features], train_df['Target_class'])

# Prediction
probs = model.predict_proba(test_df[features])
# Условие: если сигнал сильный, входим в сделку
test_df['Signal'] = np.where(probs[:, 1] > 0.65, 1, np.where(probs[:, 0] > 0.65, -1, 0))

# --- BACKTESTING ENGINE (Fix-Time Hold) ---
test_df['Entry_Signal'] = test_df['Signal'].shift(1)
test_df['Return'] = test_df['close'].diff().shift(-1)

# Логика: создаем колонку активной позиции
active_pos = np.zeros(len(test_df))
pos_counter = 0

for i in range(len(test_df)):
    if pos_counter > 0:
        active_pos[i] = current_signal
        pos_counter -= 1
    elif test_df['Entry_Signal'].iloc[i] != 0:
        current_signal = test_df['Entry_Signal'].iloc[i]
        active_pos[i] = current_signal
        pos_counter = N_BARS_HOLD - 1
    else:
        active_pos[i] = 0

test_df['Active_Pos'] = active_pos
test_df['Equity_Return'] = test_df['Active_Pos'] * test_df['Return']

# Фильтр выходных
test_df = test_df[~test_df.index.dayofweek.isin([5, 6])]
test_df['Equity'] = test_df['Equity_Return'].cumsum()

# Plot
plt.figure(figsize=(14, 6))
plt.plot(test_df.index, test_df['Equity'], label=f'Equity (Hold {N_BARS_HOLD} bars)', color='blue', linewidth=2)
plt.title(f'Fixed Hold Strategy: {symbol}, N={N_BARS_HOLD}')
plt.grid(True)
plt.legend()
plt.show()

mt5.shutdown()
