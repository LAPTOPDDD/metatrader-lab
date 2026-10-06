import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def optimize_ma_rsi_strategy():
    if not mt5.initialize():
        print("Ошибка инициализации MT5")
        return

    symbol = "EURUSD"
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3000)
    mt5.shutdown()

    if rates is None: return

    df = pd.DataFrame(rates)
    df['rsi'] = rsi(df['close'], period=14)
    returns = df['close'].pct_change()

    best_sharpe = -np.inf
    best_params = (12, 26, 30, 70)

    # Оптимизация MA + RSI фильтр (вход только в тренде + зона RSI)
    for fast in range(5, 20):
        for slow in range(20, 60):
            if fast >= slow: continue
            for rsi_low in [30, 40]:
                for rsi_high in [60, 70]:
                    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
                    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
                    
                    # Логика: BUY если MA перекрест и RSI не перекуплен, SELL если MA перекрест и RSI не перепродан
                    signal = np.zeros(len(df))
                    buy_cond = (ema_fast > ema_slow) & (df['rsi'] < rsi_high)
                    sell_cond = (ema_fast < ema_slow) & (df['rsi'] > rsi_low)
                    
                    signal[buy_cond] = 1
                    signal[sell_cond] = -1
                    
                    pnl = pd.Series(signal).shift(1) * returns
                    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252)
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = (fast, slow, rsi_low, rsi_high)

    f, s, rl, rh = best_params
    ema_fast = df['close'].ewm(span=f, adjust=False).mean()
    ema_slow = df['close'].ewm(span=s, adjust=False).mean()
    signal = np.zeros(len(df))
    signal[(ema_fast > ema_slow) & (df['rsi'] < rh)] = 1
    signal[(ema_fast < ema_slow) & (df['rsi'] > rl)] = -1
    
    df['pnl'] = pd.Series(signal).shift(1) * returns
    equity = (1 + df['pnl'].fillna(0)).cumprod()
    
    print(f"Оптимальные (MA+RSI): Fast={f}, Slow={s}, RSI_Low={rl}, RSI_High={rh}")
    print(f"Итоговая доходность: {(equity.iloc[-1]-1)*100:.2f}%")
    print(f"Коэф. Шарпа: {best_sharpe:.4f}")

if __name__ == "__main__":
    optimize_ma_rsi_strategy()
