import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

# 1. Fetch more data
symbol = "EURUSD"
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Feature Engineering
def calculate_macd(data, fast=12, slow=26):
    return data.ewm(span=fast, adjust=False).mean() - data.ewm(span=slow, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- НОВЫЕ ПРИЗНАКИ ---
def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).rolling(window=period).mean()

df['MACD'] = calculate_macd(df['close'])
df['RSI'] = calculate_rsi(df['close'])
df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])

# --- БИНАРИЗАЦИЯ + ИНТЕНСИВНОСТЬ ---
df['MACD_dir'] = np.where(df['MACD'].diff() >= 0, 1, -1)
df['RSI_diff'] = df['RSI'].diff() # Добавили интенсивность изменения RSI

# Создаем признаки
features = ['MACD_dir', 'RSI_diff', 'ATR']
for i in range(1, 4):
    df[f'MACD_bin_lag_{i}'] = df['MACD_dir'].shift(i)
    features.append(f'MACD_bin_lag_{i}')

# Target (Class)
df['Gaussian_conv'] = gaussian_filter1d(df['close'], sigma=2)
df['Target_class'] = np.where(df['Gaussian_conv'].diff() > 0, 1, 0) # 1 для роста, 0 для падения

# Drop NaNs
df.dropna(inplace=True)

# Train/Test
train_df = df.iloc[:-50]
test_df = df.iloc[-50:]

X_train = train_df[features]
y_train = train_df['Target_class']
X_test = test_df[features]

# --- КЛАССИФИКАТОР С ВЕРОЯТНОСТЯМИ ---
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.01,
    num_leaves=31,
    verbosity=-1,
    random_state=42
)
model.fit(X_train, y_train)

# Предсказываем вероятность класса 1 (рост)
probs = model.predict_proba(X_test)[:, 1]

# Фильтруем: если вероятность > 0.6 -> 1, если < 0.4 -> 0, иначе - нейтрально (0.5)
test_df['Prediction_prob'] = probs
test_df['Prediction_signal'] = np.where(probs > 0.6, 1, np.where(probs < 0.4, 0, 0.5))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(test_df.index, test_df['Target_class'], label='Actual (1=up, 0=down)')
plt.step(test_df.index, test_df['Prediction_signal'], label='Filtered Signal (1, 0, 0.5 neutral)', linestyle='--', where='post')
plt.title('Improved Classification: ATR + Probability Filtering')
plt.legend()
plt.show()

mt5.shutdown()
