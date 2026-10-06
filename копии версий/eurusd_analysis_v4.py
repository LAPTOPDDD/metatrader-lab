import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

# --- Имя файла с индексом ---
timestamp = int(time.time())
filename = f"D:\\PYTHON\\METATRADER\\eurusd_analysis_{timestamp}.py"

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

# Fetch data (увеличим до 4000 для лучшего контекста)
symbol = "EURUSD"
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 4000)
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

# --- ПРИЗНАКИ КОНТЕКСТА ---
df['SMA_10'] = df['close'].rolling(10).mean()
df['SMA_50'] = df['close'].rolling(50).mean()
df['Price_vs_SMA50'] = df['close'] / df['SMA_50'] # Отклонение от тренда
df['SMA10_vs_SMA50'] = df['SMA_10'] / df['SMA_50'] # Сила тренда

df['MACD'] = calculate_macd(df['close'])
df['RSI'] = calculate_rsi(df['close'])
df['MACD_dir'] = np.where(df['MACD'].diff() >= 0, 1, -1)

features = ['MACD_dir', 'Price_vs_SMA50', 'SMA10_vs_SMA50']
for i in range(1, 4):
    df[f'MACD_bin_lag_{i}'] = df['MACD_dir'].shift(i)
    features.append(f'MACD_bin_lag_{i}')

# Target
df['Gaussian_conv'] = gaussian_filter1d(df['close'], sigma=2)
df['Target_class'] = np.where(df['Gaussian_conv'].diff() > 0, 1, -1)

df.dropna(inplace=True)

# Train/Test (150 баров на тест)
train_df = df.iloc[:-150]
test_df = df.iloc[-150:]

X_train = train_df[features]
y_train = train_df['Target_class']
X_test = test_df[features]

# --- Модель (немного усложним) ---
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=5, # Ограничим глубину, чтобы меньше переобучалась на шуме
    verbosity=-1,
    random_state=42
)
model.fit(X_train, y_train)

# --- Прогноз ---
probs = model.predict_proba(X_test)
# Фильтр уверенности 0.55 (чуть мягче, чем 0.6)
test_df['Prediction_signal'] = np.where(probs[:, 1] > 0.55, 1, np.where(probs[:, 0] > 0.55, -1, 0))

# Plot
plt.figure(figsize=(14, 6))
plt.plot(test_df.index, test_df['Target_class'], label='Actual Class (1/-1)')
plt.step(test_df.index, test_df['Prediction_signal'], label='Prediction Signal (1/-1, 0 neutral)', linestyle='--', where='post')
plt.title(f'Analysis {timestamp}: Added Regime Features')
plt.legend()
plt.show()

mt5.shutdown()
