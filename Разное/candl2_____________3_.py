import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mt5.initialize()
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 15000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df['hour'] = df.index.hour
df['date'] = df.index.date

# Считаем азиатский диапазон (00:00 - 07:00) для каждого дня
asia = df[df['hour'] < 8].groupby('date').agg(asia_high=('high','max'), asia_low=('low','min'))
df = df.join(asia, on='date')

# Сигнал: в 08:00 (Лондон) смотрим пробой
df['signal'] = 0
london_open = df[df['hour'] == 8].copy()
london_open['signal'] = np.where(london_open['close'] > london_open['asia_high'], 1,
                          np.where(london_open['close'] < london_open['asia_low'], -1, 0))

# Держим позицию до конца дня (23:00), считаем P&L
london_open['entry'] = london_open['close']
london_open['exit'] = df[df['hour'] == 22]['close'].reindex(london_open.index, method='ffill').shift(-14)
# Проще: берем закрытие в 22:00 того же дня
close_22 = df[df['hour'] == 22].reset_index()
close_22['date'] = close_22['time'].dt.date
close_22 = close_22.set_index('date')['close']

london_open['date'] = london_open.index.date
london_open['exit'] = london_open['date'].map(close_22)
london_open['pnl'] = (london_open['exit'] - london_open['entry']) / london_open['entry'] * london_open['signal']

london_open = london_open.dropna(subset=['pnl'])
equity = (1 + london_open['pnl']).cumprod()

print(f"Всего сделок: {len(london_open[london_open['signal']!=0])}")
print(f"Win rate: {(london_open['pnl']>0).sum() / (london_open['signal']!=0).sum() * 100:.1f}%")
print(f"Итоговая доходность: {(equity.iloc[-1]-1)*100:.2f}%")

plt.figure(figsize=(12,5))
plt.plot(equity.values)
plt.title("London Breakout EURUSD H1 — Equity")
plt.grid()
plt.show()