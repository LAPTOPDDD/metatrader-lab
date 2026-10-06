import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA ====================
symbol = "EURUSD"
mt5.initialize()
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 13000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df = df[~df.index.dayofweek.isin([5, 6])]

SIGMA = 2
LOOKAHEAD = int(4 * SIGMA)
PIP = 0.0001
LOT = 100000
INIT_BALANCE = 10000

# ==================== FEATURES (КАУЗАЛЬНЫЕ) ====================
def add_features(d):
    d = d.copy()
    d['MACD'] = d['close'].ewm(span=12).mean() - d['close'].ewm(span=26).mean()
    for i in range(1, 6):
        d[f'MACD_lag_{i}'] = d['MACD'].shift(i)
    for k in [3, 6, 12, 24]:
        d[f'mom_{k}'] = d['close'].pct_change(k)
    delta = d['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
    rs = gain / loss.replace(0, np.nan)
    d['rsi_14'] = 100 - 100 / (1 + rs)
    tr = np.maximum(d['high'] - d['low'],
                    np.maximum(abs(d['high'] - d['close'].shift()),
                               abs(d['low'] - d['close'].shift())))
    d['atr_14'] = tr.ewm(alpha=1/14).mean()
    for span in [20, 50, 100]:
        d[f'ema{span}_pos'] = d['close'] / d['close'].ewm(span=span).mean() - 1
    for k in [10, 20]:
        d[f'vol_{k}'] = d['close'].rolling(k).std()
    rng = (d['high'] - d['low']).replace(0, np.nan)
    d['body'] = (d['close'] - d['open']) / rng
    d['upper_shadow'] = (d['high'] - np.maximum(d['open'], d['close'])) / rng
    d['lower_shadow'] = (np.minimum(d['open'], d['close']) - d['low']) / rng
    d['hl_range'] = (d['high'] - d['low']) / d['close']
    d['close_open'] = (d['close'] - d['open']) / d['close']
    d['vol_norm'] = d['tick_volume'] / d['tick_volume'].rolling(20).mean()
    return d

df = add_features(df)
feats = [c for c in df.columns if c not in ('time', 'open', 'high', 'low', 'close',
                                            'tick_volume', 'spread', 'real_volume',
                                            'bar_idx', 'Conv_Deriv', 'Target_Binary')]
df['bar_idx'] = np.arange(len(df))

# ==================== ОДИН ПРОГОН МОДЕЛЕЙ: получаем вероятности на тесте ====================
M = 5
test_start = len(df) - 50
test_df = df.iloc[test_start:].copy()

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

model = build_model(test_start - LOOKAHEAD)
test_df['P_long'] = np.nan
for i in range(len(test_df)):
    global_pos = test_start + i
    if i % M == 0 and i > 0:
        model = build_model(global_pos - LOOKAHEAD)
    row = df.iloc[global_pos]
    proba = model.predict_proba(row[feats].to_frame().T)[0]
    test_df.loc[test_df.index[i], 'P_long'] = proba[1]

p1 = test_df['P_long'].values
close = test_df['close'].values
high = test_df['high'].values
low = test_df['low'].values

# ==================== БЭКТЕСТ С СТОП-ЛОССОМ ====================
def backtest(N, thr, SL):
    """SL в пунктах (None = без стопа)."""
    balance = INIT_BALANCE
    pos = None
    entry = 0
    held = 0
    n_trades = 0
    wins = 0
    equity = np.empty(len(test_df))

    for i in range(len(test_df)):
        pred = 1 if p1[i] >= thr else (-1 if p1[i] <= 1 - thr else 0)
        price = close[i]
        just_closed = False

        if pos is not None:
            held += 1
            # --- Стоп-лосс по интрабар high/low ---
            if SL is not None:
                if pos == 1:
                    stop = entry - SL * PIP
                    if low[i] <= stop:
                        balance += (stop - entry) * pos * LOT
                        pos, held = None, 0
                        just_closed = True
                else:
                    stop = entry + SL * PIP
                    if high[i] >= stop:
                        balance += (stop - entry) * pos * LOT
                        pos, held = None, 0
                        just_closed = True
            # --- Таймаут по N барам ---
            if pos is not None and held >= N:
                pnl = (price - entry) * pos * LOT
                balance += pnl
                n_trades += 1
                wins += 1 if pnl > 0 else 0
                pos, held = None, 0
                just_closed = True

        # Открытие — только на следующем баре после закрытия
        if pos is None and not just_closed:
            if pred == 1:
                pos, entry, held = 1, price, 0
            elif pred == -1:
                pos, entry, held = -1, price, 0

        if pos is not None:
            equity[i] = balance + (price - entry) * pos * LOT
        else:
            equity[i] = balance

    if pos is not None:
        pnl = (close[-1] - entry) * pos * LOT
        balance += pnl
        n_trades += 1
        wins += 1 if pnl > 0 else 0

    dd = equity.min() - INIT_BALANCE
    return balance, n_trades, (wins / n_trades if n_trades else 0), dd

# ==================== ОПТИМИЗАЦИЯ: N, thr, SL ====================
print("=== Оптимизация стоп-лосса (M=5, вероятности посчитаны один раз) ===")
print(f"{'N':>3} {'thr':>5} {'SL(pips)':>9} | {'PnL':>9} {'trades':>6} {'win%':>6} {'maxDD':>9}")
results = []
for N in [3, 5, 10]:
    for thr in [0.5, 0.55, 0.6]:
        for SL in [None, 10, 20, 40, 80, 160]:
            bal, ntr, wr, dd = backtest(N, thr, SL)
            sl_txt = 'None' if SL is None else str(SL)
            print(f"{N:>3} {thr:>5.2f} {sl_txt:>9} | ${bal-10000:>8,.0f} {ntr:>6} {wr:>6.2f} ${dd:>8,.0f}")
            results.append((bal - 10000, N, thr, SL, ntr, wr, dd))

results.sort(key=lambda r: r[0], reverse=True)
print("\n=== TOP 5 ===")
for pnl, N, thr, SL, ntr, wr, dd in results[:5]:
    sl_txt = 'None' if SL is None else f'{SL} pips'
    print(f"N={N} thr={thr:.2f} SL={sl_txt} | PnL=${pnl:,.0f} trades={ntr} win%={wr:.2f} maxDD=${dd:,.0f}")

# ==================== ГРАФИК ДЛЯ ЛУЧШЕЙ КОНФИГУРАЦИИ ====================
best_pnl, best_N, best_thr, best_SL, _, _, _ = results[0]
best_bal, best_ntr, best_wr, best_dd = backtest(best_N, best_thr, best_SL)

# Пересчитаем equity лучшей конфигурации для графика
balance = INIT_BALANCE
pos = None
entry = 0
held = 0
equity = np.empty(len(test_df))
position = np.zeros(len(test_df))
for i in range(len(test_df)):
    pred = 1 if p1[i] >= best_thr else (-1 if p1[i] <= 1 - best_thr else 0)
    price = close[i]
    just_closed = False
    if pos is not None:
        held += 1
        if best_SL is not None:
            if pos == 1:
                stop = entry - best_SL * PIP
                if low[i] <= stop:
                    balance += (stop - entry) * pos * LOT
                    pos, held = None, 0
                    just_closed = True
            else:
                stop = entry + best_SL * PIP
                if high[i] >= stop:
                    balance += (stop - entry) * pos * LOT
                    pos, held = None, 0
                    just_closed = True
        if pos is not None and held >= best_N:
            pnl = (price - entry) * pos * LOT
            balance += pnl
            pos, held = None, 0
            just_closed = True
    if pos is None and not just_closed:
        if pred == 1:
            pos, entry, held = 1, price, 0
        elif pred == -1:
            pos, entry, held = -1, price, 0
    position[i] = pos if pos is not None else 0
    equity[i] = balance + (price - entry) * pos * LOT if pos is not None else balance

if pos is not None:
    balance += (close[-1] - entry) * pos * LOT

# ==================== СОХРАНЕНИЕ ЭКВИТИ В ФАЙЛ ====================
sl_txt_f = 'None' if best_SL is None else f'{best_SL}'
out_df = pd.DataFrame({
    'time': test_df.index,
    'close': close,
    'position': position,
    'P_long': p1,
    'equity': equity,
})
out_file = f'ver_2_equity_N{best_N}_thr{best_thr:.2f}_SL{sl_txt_f}.csv'
out_df.to_csv(out_file, index=False)
print(f"\nЭквити сохранено: {out_file}")

plot_df = df.tail(250)
test_plot_df = test_df.tail(100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

ax1.step(test_plot_df['bar_idx'], position[-50:], where='post',
         color='navy', linewidth=2, label=f'Position (N={best_N}, SL={best_SL})')
ax1.set_yticks([-1, 0, 1])
ax1.set_yticklabels(['Short', 'Flat', 'Long'])
sl_txt = 'None' if best_SL is None else f'{best_SL} pips'
ax1.set_title(f'ver_2 Position / Signals (M=5, N={best_N}, thr={best_thr:.2f}, SL={sl_txt})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.step(test_plot_df['bar_idx'], equity[-50:], where='post',
         color='green', linewidth=2, label='Equity')
ax2.axhline(y=INIT_BALANCE, color='gray', linestyle='--', alpha=0.5)
ax2.set_title(f'Equity (Final \${balance:,.0f}, trades={best_ntr}, win%={best_wr:.2f}, maxDD=\${best_dd:,.0f})')
ax2.set_ylabel('Balance ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

step = max(1, len(plot_df) // 12)
tick_pos = plot_df['bar_idx'].iloc[::step].values
tick_lbl = plot_df.index.to_series().dt.strftime('%d-%m %H:%M').iloc[::step].values
for ax in (ax1, ax2):
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, fontsize=9)

plt.tight_layout()
plt.show()

mt5.shutdown()
