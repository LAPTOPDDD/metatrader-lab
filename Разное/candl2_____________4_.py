import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mt5.initialize()

def get_data(sym):
    r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 15000)
    d = pd.DataFrame(r)
    d['time'] = pd.to_datetime(d['time'], unit='s')
    return d.set_index('time')['close']

eur = get_data("EURUSD")
gbp = get_data("GBPUSD")
df = pd.concat([eur, gbp], axis=1).dropna()
df.columns = ['EUR', 'GBP']

# Beta через скользящую регрессию (упрощенно — отношение)
window = 200
df['spread'] = np.log(df['EUR']) - np.log(df['GBP'])
df['mean'] = df['spread'].rolling(window).mean()
df['std'] = df['spread'].rolling(window).std()
df['zscore'] = (df['spread'] - df['mean']) / df['std']

# Сигналы
df['pos'] = 0
df.loc[df['zscore'] > 2, 'pos'] = -1   # спред слишком высок → шорт спреда
df.loc[df['zscore'] < -2, 'pos'] = 1    # спред слишком низок → лонг спреда
df.loc[df['zscore'].abs() < 0.5, 'pos'] = 0  # закрываем при возврате
df['pos'] = df['pos'].replace(0, np.nan).ffill().fillna(0)

# P&L: доход от спреда
df['spread_ret'] = df['spread'].diff()
df['strategy'] = df['pos'].shift(1) * df['spread_ret']

df = df.dropna()
equity = df['strategy'].cumsum()  # спред уже в логах, суммируем

trades = (df['pos'].diff() != 0).sum()
print(f"Всего смен позиции: {trades}")
print(f"Итоговый P&L (в лог-пунктах спреда): {equity.iloc[-1]:.4f}")
print(f"Sharpe: {df['strategy'].mean() / df['strategy'].std() * np.sqrt(24*252):.2f}")

plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(df['zscore'])
plt.axhline(2, color='r', ls='--'); plt.axhline(-2, color='g', ls='--'); plt.axhline(0, color='k')
plt.title("Z-score спреда EURUSD-GBPUSD")
plt.subplot(2,1,2)
plt.plot(equity.values)
plt.title("Equity парного трейдинга")
plt.tight_layout()
plt.show()