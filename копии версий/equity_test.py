import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Setup
symbol = "EURUSD"
N_BARS_HOLD = 4
if not mt5.initialize(): quit()

# 1. Fetch & Filter
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df = df[~df.index.dayofweek.isin([5, 6])].copy()

# 2. Indicators
df['Conv'] = gaussian_filter1d(df['close'].values, sigma=12)
df['Conv_Diff'] = df['Conv'].diff()
df['Target_Binary'] = np.where(df['Conv_Diff'] > 0, 1, -1)

# Features
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6): df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]
df_clean = df.dropna().copy()

# 3. Train
train_df = df_clean.iloc[:-100]
test_df = df_clean.iloc[-100:].copy()

model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
model.fit(train_df[features], train_df['Target_Binary'])

# 4. Predict & Backtest
model.fit(train_df[features], train_df['Target_Binary'])
test_df['Prediction'] = model.predict(test_df[features])

# Print сигналы для проверки
print(test_df['Prediction'].head(10)) 

# Логика: если Prediction > 0 -> Buy (1), если < 0 -> Sell (-1)
# Вход на открытии следующего бара (сдвигаем сигнал)
test_df['Entry_Signal'] = test_df['Prediction'].shift(1)

# Доходность следующего бара
test_df['Return'] = test_df['close'].diff().shift(-1).fillna(0)

# Backtest engine
equity_ret = np.zeros(len(test_df))
active_pos, pos_counter = 0, 0

for i in range(len(test_df)):
    if active_pos != 0:
        pos_counter -= 1
        equity_ret[i] = active_pos * test_df['Return'].iloc[i]
        if pos_counter <= 0: active_pos = 0
    elif test_df['Entry_Signal'].iloc[i] != 0:
        active_pos = test_df['Entry_Signal'].iloc[i]
        pos_counter = N_BARS_HOLD
        equity_ret[i] = active_pos * test_df['Return'].iloc[i]

test_df['Equity'] = equity_ret.cumsum()

# 5. Plot Equity
plt.figure(figsize=(14, 6))
plt.plot(test_df.index, test_df['Equity'], label=f'Equity (Hold {N_BARS_HOLD} bars)', color='blue', linewidth=2)
plt.title(f'Equity Curve: Approximator Signals, Hold={N_BARS_HOLD}')
plt.grid(True)
plt.legend()
plt.show()

mt5.shutdown()
