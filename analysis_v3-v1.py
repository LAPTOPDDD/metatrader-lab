import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==================== SETUP ====================
symbol = "EURUSD"
if not mt5.initialize():
    quit()

# ==================== DATA ====================
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Удаляем выходные (не показываем суб/вс на графике)
df = df[~df.index.dayofweek.isin([5, 6])]

# ==================== ПЕРВАЯ ПРОИЗВОДНАЯ ГАУССА ====================
SIGMA = 2
LOOKAHEAD = int(4 * SIGMA)  # радиус окна конволюции — на столько баров в будущее она "видит"
df['Conv_Deriv'] = gaussian_filter1d(df['close'], SIGMA, order=1)

# ==================== TARGET (БИНАРНАЯ ФОРМА) ====================
df['Target_Binary'] = np.where(df['Conv_Deriv'] > 0, 1, -1)

# ==================== FEATURES (ВСЕ КАУЗАЛЬНЫЕ — ТОЛЬКО ПРОШЛОЕ) ====================
def add_features(d):
    d = d.copy()
    # MACD + лаги
    d['MACD'] = d['close'].ewm(span=12).mean() - d['close'].ewm(span=26).mean()
    for i in range(1, 6):
        d[f'MACD_lag_{i}'] = d['MACD'].shift(i)
    # Моментум (прошлое)
    for k in [3, 6, 12, 24]:
        d[f'mom_{k}'] = d['close'].pct_change(k)
    # RSI(14)
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
    # Свечные паттерны
    rng = (d['high'] - d['low']).replace(0, np.nan)
    d['body'] = (d['close'] - d['open']) / rng
    d['upper_shadow'] = (d['high'] - np.maximum(d['open'], d['close'])) / rng
    d['lower_shadow'] = (np.minimum(d['open'], d['close']) - d['low']) / rng
    d['hl_range'] = (d['high'] - d['low']) / d['close']
    d['close_open'] = (d['close'] - d['open']) / d['close']
    # Объём (нормированный)
    d['vol_norm'] = d['tick_volume'] / d['tick_volume'].rolling(20).mean()
    return d

df = add_features(df)

feats = [c for c in df.columns if c not in ('time', 'open', 'high', 'low', 'close',
                                            'tick_volume', 'spread', 'real_volume',
                                            'bar_idx', 'Conv_Deriv', 'Target_Binary')]

# ==================== CLEAN DATA ====================
# NaN (начало из-за rolling/ewm) оставляем — убираются внутри build_model при обучении;
# глобально не дропаем, чтобы не сдвигать окно теста

# Плотный бар-индекс (без дырок от выходных) — для графика
df['bar_idx'] = np.arange(len(df))

# ==================== MODEL (ROLLING RETRAIN КАЖДЫЕ M БАРОВ) ====================
M = 5           # каждые M тестовых баров переобучаем
N = 3           # удержание позиции
THRESH = 0.50   # порог уверенности: входим если P(side) >= THRESH
INIT_BALANCE = 10000

test_start = len(df) - 100
test_df = df.iloc[test_start:].copy()


def build_model(end_excl):
    """Обучает модель на данных df.iloc[:end_excl].

    Отступ от края на LOOKAHEAD уже учтён на вызывающей стороне,
    поэтому производная на последних барах не заглядывает в будущее.
    """
    sub = add_features(df.iloc[:end_excl])
    sub['Conv_Deriv'] = gaussian_filter1d(sub['close'].values, SIGMA, order=1)
    sub['Target_Binary'] = np.where(sub['Conv_Deriv'] > 0, 1, -1)
    sub.dropna(inplace=True)

    m = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        verbosity=-1,
        random_state=42
    )
    m.fit(sub[feats], sub['Target_Binary'])
    return m


# Изначальное обучение: отступаем от начала теста на LOOKAHEAD,
# чтобы производная на границе не видела будущие (тестовые) бары
model = build_model(test_start - LOOKAHEAD)

test_df['Prediction'] = np.nan
balance = INIT_BALANCE
pos = None
entry_price = 0
bars_held = 0
n_retrains = 1

for i in range(len(test_df)):
    global_pos = test_start + i

    # Каждые M баров — переобучение с отступом LOOKAHEAD от края (нет взгляда в будущее)
    if i % M == 0 and i > 0:
        model = build_model(global_pos - LOOKAHEAD)
        n_retrains += 1

    row = df.iloc[global_pos]
    proba = model.predict_proba(row[feats].to_frame().T)[0]
    p1 = proba[1]
    pred = 1 if p1 >= THRESH else (-1 if p1 <= 1 - THRESH else 0)
    test_df.loc[test_df.index[i], 'Prediction'] = pred
    price = row['close']

    # ---------- Торговля: удержание N баров ----------
    just_closed = False
    if pos is not None:
        bars_held += 1
        if bars_held >= N:
            balance += (price - entry_price) * pos * 100000  # 1 лот
            pos = None
            bars_held = 0
            just_closed = True  # не открываем новый вход на том же баре

    # Открытие новой позиции ТОЛЬКО на следующем баре после закрытия
    if pos is None and not just_closed:
        if pred == 1:
            pos = 1
            entry_price = price
            bars_held = 0
        elif pred == -1:
            pos = -1
            entry_price = price
            bars_held = 0

    # Mark-to-market эквити
    if pos is not None:
        test_df.loc[test_df.index[i], 'equity'] = balance + (price - entry_price) * pos * 100000
    else:
        test_df.loc[test_df.index[i], 'equity'] = balance

# Закрываем остаток в конце
if pos is not None:
    price = test_df.iloc[-1]['close']
    balance += (price - entry_price) * pos * 100000
    test_df.loc[test_df.index[-1], 'equity'] = balance

print(f"\nRetrain every {M} bars: {n_retrains} fits, LOOKAHEAD={LOOKAHEAD} (4*sigma)")
print(f"Backtest N={N} thr={THRESH}: Final equity = ${balance:,.2f} (PnL ${balance - INIT_BALANCE:,.2f})")

# ==================== PLOT ====================
plot_df = df.tail(250)
test_plot_df = test_df.tail(100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

# ---------- ГРАФИК 1: Таргет vs Предсказание ----------
ax1.step(plot_df['bar_idx'], plot_df['Target_Binary'], where='post',
         color='darkblue', linewidth=2, alpha=0.7, label='Actual Target')
ax1.step(test_plot_df['bar_idx'], test_plot_df['Prediction'], where='post',
         color='orange', linewidth=2, linestyle='--', label='LGBM Prediction')
ax1.set_yticks([-1, 0, 1])
ax1.set_yticklabels(['Short', 'Flat', 'Long'])
ax1.set_title(f'Target (Binary 1st Deriv of Gaussian) vs LGBM Prediction (M={M}, thr={THRESH})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ---------- ГРАФИК 2: Эквити ----------
ax2.step(test_plot_df['bar_idx'], test_plot_df['equity'], where='post',
         color='green', linewidth=2, label=f'Equity (N={N} bars)')
ax2.axhline(y=INIT_BALANCE, color='gray', linestyle='--', alpha=0.5)
ax2.set_title(f'Equity Curve (N={N}, Start ${INIT_BALANCE:,}, Final ${balance:,.0f})')
ax2.set_ylabel('Balance ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Подписи оси X — реальные даты на плотном индексе (выходные скрыты)
step = max(1, len(plot_df) // 12)
tick_pos = plot_df['bar_idx'].iloc[::step].values
tick_lbl = plot_df.index.to_series().dt.strftime('%d-%m %H:%M').iloc[::step].values
for ax in (ax1, ax2):
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, fontsize=9)

plt.tight_layout()
plt.show()

# ==================== METRICS ====================
from sklearn.metrics import accuracy_score, classification_report
active = test_df[test_df['Prediction'] != 0]
acc = accuracy_score(active['Target_Binary'], active['Prediction'])
print(f"\nModel Accuracy on Active Signals (trades): {acc:.4f}  ({len(active)} of {len(test_df)} bars)")
if len(active) > 0:
    print("\nClassification Report (active signals):")
    print(classification_report(active['Target_Binary'], active['Prediction'], target_names=['Short (-1)', 'Long (1)']))

mt5.shutdown()
