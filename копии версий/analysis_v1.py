import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Setup
symbol = "EURUSD"
if not mt5.initialize(): quit()

# Fetch enough data for training + test (last 100) + context
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 12000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# 1. Indicator Calculation - First Derivative of Gaussian as Target
df['Target_Signal'] = gaussian_filter1d(df['close'], sigma=2, order=1)
df['Target_Binary'] = np.where(df['Target_Signal'] > 0, 1, -1)

# Features (MACD, RSI)
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]

# Drop NaNs
df.dropna(inplace=True)

# Train (Everything before last 100) / Test (Last 100)
train_df = df.iloc[:-100]
test_df = df.iloc[-100:].copy()

# 2. Train Approximator
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
model.fit(train_df[features], train_df['Target_Binary'])

# Predict
test_df['Prediction'] = model.predict(test_df[features])

# 3. Backtest: hold N bars per signal
N = 12
equity = [10000]
equity_idx = [test_df.index[0]]
balance = 10000
pos = None
entry_price = 0
bars_held = 0

for i in range(len(test_df)):
    row = test_df.iloc[i]
    sig = row['Prediction']
    price = row['close']
    
    # Close position after N bars
    if pos is not None:
        bars_held += 1
        if bars_held >= N:
            pnl = (price - entry_price) * pos * 100000  # 1 lot
            balance += pnl
            equity.append(balance)
            equity_idx.append(row.name)
            pos = None
            bars_held = 0
    # Open new position
    if pos is None:
        if sig == 1:
            pos = 1  # buy
            entry_price = price
            bars_held = 0
        elif sig == -1:
            pos = -1  # sell
            entry_price = price
            bars_held = 0

# Close any remaining position
if pos is not None:
    pnl = (test_df.iloc[-1]['close'] - entry_price) * pos * 100000
    balance += pnl
    equity.append(balance)
    equity_idx.append(test_df.index[-1])

# Plot (Last 200 bars)
plot_df = df.tail(200)
test_plot_df = test_df.tail(100)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

# Plot 1: Close Price
ax1.plot(plot_df.index, plot_df['close'], label='Close Price', color='black')
ax1.set_title('EURUSD H1: Close Price')
ax1.legend()

# Plot 2: First Derivative of Gaussian
ax2.plot(plot_df.index, plot_df['Target_Signal'], label='1st Derivative Gaussian', color='red')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('1st Derivative of Gaussian (Target Signal)')
ax2.legend()

# Plot 3: Binary Target vs Approximator
ax3.step(test_plot_df.index, test_plot_df['Target_Binary'], label='Target (1st Deriv Sign)', where='post', alpha=0.5)
ax3.step(test_plot_df.index, test_plot_df['Prediction'], label='LGBM Approximator', where='post', linestyle='--')
ax3.set_title('Target vs. LightGBM Approximator (Test Area - Last 100 Bars)')
ax3.legend()

# Plot 4: Equity Curve
ax4.plot(equity_idx, equity, label='Equity', color='green', linewidth=1.5)
ax4.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
ax4.set_title(f'Equity Curve (Hold N={N} bars, Start $10,000)')
ax4.set_ylabel('Balance ($)')
ax4.legend()

plt.tight_layout()
plt.show()

mt5.shutdown()
