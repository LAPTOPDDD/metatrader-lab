import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA ====================
symbol = "EURUSD"
mt5.initialize()
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df = df[~df.index.dayofweek.isin([5, 6])]

SIGMA = 2
LOOKAHEAD = int(4 * SIGMA)

def add_features(d):
    d = d.copy()
    # MACD + лаги
    d['MACD'] = d['close'].ewm(span=12).mean() - d['close'].ewm(span=26).mean()
    for i in range(1, 6):
        d[f'MACD_lag_{i}'] = d['MACD'].shift(i)
    # Моментум (прошлое)
    for k in [3, 6, 12, 24]:
        d[f'mom_{k}'] = d['close'].pct_change(k)
    # RSI(14) — каузальный
    delta = d['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
    rs = gain / loss.replace(0, np.nan)
    d['rsi_14'] = 100 - 100 / (1 + rs)
    # ATR(14)
    tr = np.maximum(d['high'] - d['low'],
                    np.maximum(abs(d['high'] - d['close'].shift()),
                               abs(d['low'] - d['close'].shift())))
    d['atr_14'] = tr.ewm(alpha=1/14).mean()
    # Положение цены относительно EMA
    for span in [20, 50, 100]:
        d[f'ema{span}_pos'] = d['close'] / d['close'].ewm(span=span).mean() - 1
    # Волатильность
    for k in [10, 20]:
        d[f'vol_{k}'] = d['close'].rolling(k).std()
    # Свечные паттерны (каузальные)
    rng = (d['high'] - d['low']).replace(0, np.nan)
    d['body'] = (d['close'] - d['open']) / rng
    d['upper_shadow'] = (d['high'] - np.maximum(d['open'], d['close'])) / rng
    d['lower_shadow'] = (np.minimum(d['open'], d['close']) - d['low']) / rng
    d['hl_range'] = (d['high'] - d['low']) / d['close']
    d['close_open'] = (d['close'] - d['open']) / d['close']
    # Объём (если есть) — нормированный
    if 'tick_volume' in d.columns and d['tick_volume'].abs().sum() > 0:
        d['vol_norm'] = d['tick_volume'] / d['tick_volume'].rolling(20).mean()
    else:
        d['vol_norm'] = 0.0
    return d

df = add_features(df)

feats = [c for c in df.columns if c not in ('time', 'open', 'high', 'low', 'close',
                                             'tick_volume', 'spread', 'real_volume',
                                             'bar_idx', 'Conv_Deriv', 'Target_Binary')]

def build_model(end_excl):
    sub = add_features(df.iloc[:end_excl])
    sub['Conv_Deriv'] = gaussian_filter1d(sub['close'].values, SIGMA, order=1)
    sub['Target_Binary'] = np.where(sub['Conv_Deriv'] > 0, 1, -1)
    sub.dropna(inplace=True)
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03,
                           num_leaves=31, min_child_samples=20,
                           verbosity=-1, random_state=42)
    m.fit(sub[feats], sub['Target_Binary'])
    return m

test_start = len(df) - 100

def backtest(M, N, thresh):
    model = build_model(test_start - LOOKAHEAD)
    balance = 10000
    pos = None
    entry_price = 0
    bars_held = 0
    n_trades = 0
    wins = 0

    for i in range(len(test_df)):
        global_pos = test_start + i
        if i % M == 0 and i > 0:
            model = build_model(global_pos - LOOKAHEAD)
        row = df.iloc[global_pos]
        f = row[feats].to_frame().T
        proba = model.predict_proba(f)[0]
        p1 = proba[1]
        price = row['close']

        # Закрытие
        if pos is not None:
            bars_held += 1
            if bars_held >= N:
                pnl = (price - entry_price) * pos * 100000
                balance += pnl
                n_trades += 1
                wins += 1 if pnl > 0 else 0
                pos = None
                bars_held = 0
        # Открытие
        if pos is None:
            if p1 >= thresh:
                pos = 1
                entry_price = price
                bars_held = 0
            elif p1 <= 1 - thresh:
                pos = -1
                entry_price = price
                bars_held = 0

    if pos is not None:
        price = test_df.iloc[-1]['close']
        pnl = (price - entry_price) * pos * 100000
        balance += pnl
        n_trades += 1
        wins += 1 if pnl > 0 else 0

    return balance, n_trades, (wins / n_trades if n_trades else 0)

test_df = df.iloc[test_start:].copy()
results = []
for M in [5, 10, 20]:
    for N in [3, 5, 10]:
        for thresh in [0.5, 0.6]:
            bal, ntr, wr = backtest(M, N, thresh)
            results.append((bal - 10000, bal, M, N, thresh, ntr, wr))
            print(f"M={M:3d} N={N:3d} thr={thresh:.2f} | PnL=${bal-10000:8,.2f} trades={ntr:3d} winrate={wr:.2f}")

results.sort(reverse=True)
print("\n===== TOP 5 =====")
for pnl, bal, M, N, thresh, ntr, wr in results[:5]:
    print(f"M={M} N={N} thr={thresh:.2f} | PnL=${pnl:,.2f} trades={ntr} winrate={wr:.2f}")

mt5.shutdown()
