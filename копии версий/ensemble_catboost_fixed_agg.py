import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Символы и таймфреймы
symbol = "EURUSD"
timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1]
hold_periods = {mt5.TIMEFRAME_M5: 2, mt5.TIMEFRAME_M15:2, mt5.TIMEFRAME_M30:2, mt5.TIMEFRAME_H1: 2}

if not mt5.initialize(): quit()

def get_signal_equity(tf, n_hold):
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 5000)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Features
    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['MACD_dir'] = np.where(df['MACD'].diff() >= 0, 1, -1)
    df['Gaussian_conv'] = gaussian_filter1d(df['close'], sigma=2)
    df['Target'] = np.where(df['Gaussian_conv'].diff() > 0, 1, -1)
    
    for i in range(1, 4): df[f'Lag_{i}'] = df['MACD_dir'].shift(i)
    df.dropna(inplace=True)

    train = df.iloc[:-500]
    test = df.iloc[-500:].copy()
    features = ['MACD_dir', 'Lag_1', 'Lag_2', 'Lag_3']
    
    model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    model.fit(train[features], train['Target'])
    
    probs = model.predict_proba(test[features])
    threshold = 0.85
    test['Signal'] = np.where(probs[:, 1] > threshold, 1, np.where(probs[:, 0] > threshold, -1, 0))

    test['Return'] = test['close'].diff().shift(-1).fillna(0)
    
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
            
    return pd.Series(equity_ret, index=test.index) # Возвращаем ДНЕВНУЮ ДОХОДНОСТЬ

# Собираем данные
plt.figure(figsize=(14, 8))

# Инициализируем DataFrame для суммирования
combined_returns = pd.DataFrame(index=pd.date_range(start='2026-06-01', end='2026-08-01', freq='H'))

for tf in timeframes:
    equity_ret = get_signal_equity(tf, hold_periods[tf])
    # Добавляем в общую таблицу и суммируем
    combined_returns = combined_returns.join(equity_ret.rename(f'TF_{tf}'), how='outer')
    plt.plot(equity_ret.cumsum().index, equity_ret.cumsum(), label=f'TF {tf} Equity', alpha=0.5)

# Суммарная эквити = кумулятивная сумма СУММЫ доходностей всех TF
total_equity = combined_returns.fillna(0).sum(axis=1).cumsum()

plt.plot(total_equity.index, total_equity, label='TOTAL ENSEMBLE EQUITY (CatBoost Fixed)', color='black', linewidth=3)
plt.legend()
plt.title("Ensemble Backtest: Fixed Aggregation")
plt.grid(True)
plt.show()

mt5.shutdown()
