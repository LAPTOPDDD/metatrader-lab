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

# Fetch data
symbol = "EURUSD"
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Features
def calculate_macd(data, fast=12, slow=26):
    return data.ewm(span=fast, adjust=False).mean() - data.ewm(span=slow, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['MACD'] = calculate_macd(df['close'])
df['RSI'] = calculate_rsi(df['close'])
df['MACD_dir'] = np.where(df['MACD'].diff() >= 0, 1, -1)
df['RSI_diff'] = df['RSI'].diff()

features = ['MACD_dir', 'RSI_diff']
for i in range(1, 4):
    df[f'MACD_bin_lag_{i}'] = df['MACD_dir'].shift(i)
    features.append(f'MACD_bin_lag_{i}')

# Target (Class: 1 для роста, -1 для падения)
df['Gaussian_conv'] = gaussian_filter1d(df['close'], sigma=2)
df['Target_class'] = np.where(df['Gaussian_conv'].diff() > 0, 1, -1)

df.dropna(inplace=True)

# Train/Test
train_df = df.iloc[:-150]
test_df = df.iloc[-150:]

X_train = train_df[features]
y_train = train_df['Target_class']
X_test = test_df[features]

# --- Модель ---
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    verbosity=-1,
    random_state=42
)
model.fit(X_train, y_train)

# --- Прогноз с учетом вероятностей ---
# predict_proba для -1 и 1
probs = model.predict_proba(X_test)
# prob_1 = probs[:, 1]  (вероятность класса 1)
# prob_minus1 = probs[:, 0] (вероятность класса -1)

# Условие: если уверенность в классе > 0.6, берем сигнал, иначе 0
test_df['Prediction_signal'] = np.where(probs[:, 1] > 0.6, 1, np.where(probs[:, 0] > 0.6, -1, 0))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(test_df.index, test_df['Target_class'], label='Actual Class (1/-1)')
plt.step(test_df.index, test_df['Prediction_signal'], label='Prediction Signal (1/-1, 0 neutral)', linestyle='--', where='post')
plt.title(f'Analysis {timestamp}: Symmetric Classes (-1/1)')
plt.legend()
plt.show()

mt5.shutdown()
