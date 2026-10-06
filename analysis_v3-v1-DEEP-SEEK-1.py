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
df['Conv_Deriv'] = gaussian_filter1d(df['close'], sigma=2, order=1)

# ==================== TARGET (БИНАРНАЯ ФОРМА) ====================
df['Target_Binary'] = np.where(df['Conv_Deriv'] > 0, 1, -1)

# ==================== FEATURES (MACD LAGS) ====================
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]

# ==================== CLEAN DATA ====================
df.dropna(inplace=True)

# Плотный бар-индекс (без дырок от выходных) — для графика
df['bar_idx'] = np.arange(len(df))

# ==================== TRAIN / TEST SPLIT ====================
train_df = df.iloc[:-100]
test_df = df.iloc[-100:].copy()

# ==================== MODEL ====================
model = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    verbosity=-1,
    random_state=42
)
model.fit(train_df[features], train_df['Target_Binary'])

test_df['Prediction'] = model.predict(test_df[features])

# ==================== BACKTEST: HOLD N=3 BAR ====================
N = 4
INIT_BALANCE = 10000
balance = INIT_BALANCE
pos = None
entry_price = 0
bars_held = 0

test_df['equity'] = np.nan
test_df['pos'] = 0

for i in range(len(test_df)):
    row = test_df.iloc[i]
    sig = row['Prediction']
    price = row['close']

    # Закрытие позиции после N баров
    if pos is not None:
        bars_held += 1
        if bars_held >= N:
            pnl = (price - entry_price) * pos * 100000  # 1 лот
            balance += pnl
            pos = None
            bars_held = 0

    # Открытие новой позиции
    if pos is None:
        if sig == 1:
            pos = 1
            entry_price = price
            bars_held = 0
        elif sig == -1:
            pos = -1
            entry_price = price
            bars_held = 0

    test_df.loc[row.name, 'pos'] = pos if pos is not None else 0

    # Mark-to-market эквити: если есть позиция — переоцениваем по текущей цене
    if pos is not None:
        unrealized = (price - entry_price) * pos * 100000
        test_df.loc[row.name, 'equity'] = balance + unrealized
    else:
        test_df.loc[row.name, 'equity'] = balance

# Закрываем остаток в конце
if pos is not None:
    price = test_df.iloc[-1]['close']
    pnl = (price - entry_price) * pos * 100000
    balance += pnl
    test_df.loc[test_df.index[-1], 'equity'] = balance

print(f"\nBacktest N={N}: Final equity = ${balance:,.2f} (PnL ${balance - INIT_BALANCE:,.2f})")

# ==================== PLOT ====================
plot_df = df.tail(250)
test_plot_df = test_df.tail(100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

# ---------- ГРАФИК 1: Таргет vs Предсказание ----------
ax1.step(plot_df['bar_idx'], plot_df['Target_Binary'], where='post',
         color='darkblue', linewidth=2, alpha=0.7, label='Actual Target')
ax1.step(test_plot_df['bar_idx'], test_plot_df['Prediction'], where='post',
         color='orange', linewidth=2, linestyle='--', label='LGBM Prediction')
ax1.set_yticks([-1, 1])
ax1.set_yticklabels(['Short', 'Long'])
ax1.set_title('Target (Binary 1st Deriv of Gaussian) vs LGBM Prediction (Last 100 Bars)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ---------- ГРАФИК 2: Эквити ----------
ax2.step(test_plot_df['bar_idx'], test_plot_df['equity'], where='post',
         color='green', linewidth=2, label=f'Equity (N={N} bars)')
ax2.axhline(y=INIT_BALANCE, color='gray', linestyle='--', alpha=0.5)
ax2.set_title(f'Equity Curve (Hold N={N} bars, Start ${INIT_BALANCE:,})')
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
acc = accuracy_score(test_df['Target_Binary'], test_df['Prediction'])
print(f"\nModel Accuracy on Test Set: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(test_df['Target_Binary'], test_df['Prediction'], target_names=['Short (-1)', 'Long (1)']))

mt5.shutdown()
