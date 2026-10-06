import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d

# --- ОПТИМИЗАТОР N ---
symbol = "EURUSD"
if not mt5.initialize(): quit()

rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 5000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Features
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

train_df = df.iloc[:-150]
test_df = df.iloc[-150:].copy()

model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.01, num_leaves=31, max_depth=6, verbosity=-1, random_state=42)
model.fit(train_df[features], train_df['Target_class'])

probs = model.predict_proba(test_df[features])
test_df['Signal'] = np.where(probs[:, 1] > 0.65, 1, np.where(probs[:, 0] > 0.65, -1, 0))
test_df['Entry_Signal'] = test_df['Signal'].shift(1)
test_df['Return'] = test_df['close'].diff().shift(-1)
test_df = test_df[~test_df.index.dayofweek.isin([5, 6])]

print(f"--- Тестирование оптимального N для {symbol} ---")

for n_hold in range(1, 11):
    active_pos = 0
    pos_counter = 0
    equity_returns = np.zeros(len(test_df))
    
    for i in range(len(test_df)):
        signal = test_df['Entry_Signal'].iloc[i]
        
        if active_pos != 0:
            pos_counter -= 1
            if pos_counter <= 0:
                equity_returns[i] = active_pos * test_df['Return'].iloc[i]
                active_pos = 0
            else:
                equity_returns[i] = active_pos * test_df['Return'].iloc[i]
        elif signal != 0:
            active_pos = signal
            pos_counter = n_hold
            equity_returns[i] = active_pos * test_df['Return'].iloc[i]
            
    total_profit = equity_returns.sum()
    print(f"N={n_hold}: Суммарная доходность = {total_profit:.5f}")

mt5.shutdown()
