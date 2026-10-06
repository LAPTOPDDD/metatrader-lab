import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Символы и таймфреймы
symbol = "EURUSD"
timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1]
hold_periods = {mt5.TIMEFRAME_M5: 7, mt5.TIMEFRAME_M15: 4, mt5.TIMEFRAME_M30: 3, mt5.TIMEFRAME_H1: 2}

if not mt5.initialize(): quit()

def get_signal_equity(tf, n_hold):
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 5000)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Features (упрощенные для примера)
    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['MACD_dir'] = np.where(df['MACD'].diff() >= 0, 1, -1)
    df['Gaussian_conv'] = gaussian_filter1d(df['close'], sigma=2)
    df['Target'] = np.where(df['Gaussian_conv'].diff() > 0, 1, -1)
    
    # Лаги
    for i in range(1, 4): df[f'Lag_{i}'] = df['MACD_dir'].shift(i)
    df.dropna(inplace=True)

    # Train/Test
    train = df.iloc[:-500]
    test = df.iloc[-500:].copy()
    features = ['MACD_dir', 'Lag_1', 'Lag_2', 'Lag_3']
    
    model = lgb.LGBMClassifier(verbosity=-1, random_state=42).fit(train[features], train['Target'])
    probs = model.predict_proba(test[features])
    test['Signal'] = np.where(probs[:, 1] > 0.6, 1, np.where(probs[:, 0] > 0.6, -1, 0))

    # Backtest
    test['Return'] = test['close'].diff().shift(-1).fillna(0)
    
    # Системная логика
    equity_ret = np.zeros(len(test))
    active_pos, pos_counter = 0, 0
    
    for i in range(len(test)):
        if active_pos != 0:
            pos_counter -= 1
            equity_ret[i] = active_pos * test['Return'].iloc[i]
            if pos_counter <= 0: active_pos = 0
        elif test['Signal'].iloc[i] != 0:
            active_pos = test['Signal'].iloc[i]
            pos_counter = n_hold
            equity_ret[i] = active_pos * test['Return'].iloc[i]
            
    return pd.Series(equity_ret.cumsum(), index=test.index)

# Собираем данные
plt.figure(figsize=(14, 8))
total_equity = pd.Series(0, index=pd.date_range(start='2026-01-01', periods=1, freq='H')) # Dummy

for tf in timeframes:
    equity = get_signal_equity(tf, hold_periods[tf])
    plt.plot(equity.index, equity, label=f'TF {tf} Equity', alpha=0.5)
    # Суммируем (приводим к общему индексу для простоты)
    if total_equity.index[0].year == 2026: total_equity = equity
    else: total_equity = total_equity.add(equity, fill_value=0)

plt.plot(total_equity.index, total_equity, label='TOTAL ENSEMBLE EQUITY', color='black', linewidth=3)
plt.legend()
plt.title("Ensemble Backtest: Multi-Timeframe Strategy")
plt.grid(True)
plt.show()

mt5.shutdown()
