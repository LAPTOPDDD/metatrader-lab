import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

timestamp = int(time.time())

# Initialize MT5
if not mt5.initialize(): quit()

# Data
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 4000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# --- ПРОФЕССИОНАЛЬНЫЕ ФИЧИ ---
# 1. Нормализация цены (Stochastic)
period_n = 20
df['Rel_Price'] = (df['close'] - df['low'].rolling(period_n).min()) / \
                  (df['high'].rolling(period_n).max() - df['low'].rolling(period_n).min())

# 2. Трендовый контекст
df['SMA_50'] = df['close'].rolling(50).mean()
df['Dist_SMA50'] = (df['close'] - df['SMA_50']) / df['SMA_50']

# 3. Импульс
df['Momentum'] = df['close'].diff(5)

# Индикаторы
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['MACD_dir'] = np.where(df['MACD'].diff() >= 0, 1, -1)

features = ['Rel_Price', 'Dist_SMA50', 'Momentum', 'MACD_dir']
for i in range(1, 6):
    df[f'Lag_{i}'] = df['MACD_dir'].shift(i)
    features.append(f'Lag_{i}')

# Target (Гаусс)
df['Gaussian_conv'] = gaussian_filter1d(df['close'], sigma=2)
df['Target_class'] = np.where(df['Gaussian_conv'].diff() > 0, 1, -1)

df.dropna(inplace=True)

# Train/Test
train_df = df.iloc[:-150]
test_df = df.iloc[-150:]

# Веса для обучения: если цена резко меняется (большой diff конволюции), вес выше
sample_weights = np.abs(df['Gaussian_conv'].diff().loc[train_df.index]) + 1

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=6,
    verbosity=-1,
    random_state=42
)
model.fit(train_df[features], train_df['Target_class'], sample_weight=sample_weights)

# --- Прогноз с высоким порогом уверенности (0.65) ---
probs = model.predict_proba(test_df[features])
# 1 = probs[:, 1], -1 = probs[:, 0]
test_df['Prediction_signal'] = np.where(probs[:, 1] > 0.65, 1, np.where(probs[:, 0] > 0.65, -1, 0))

# Plot
plt.figure(figsize=(14, 6))
plt.plot(test_df.index, test_df['Target_class'], label='Actual', alpha=0.7)
plt.step(test_df.index, test_df['Prediction_signal'], label='Professional Signal', linestyle='--', where='post')
plt.title(f'Analysis {timestamp}: Optimized with Weights & Norms')
plt.legend()
plt.show()

mt5.shutdown()
