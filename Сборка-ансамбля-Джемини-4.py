import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import os

# ================== НАСТРОЙКИ V4 (MASTER) ==================
SYMBOLS = [
    "USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "XAUUSD" # Включая Золото!
]
TIMEFRAMES = [
    mt5.TIMEFRAME_M30,
    mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_H4,
]
TIMEFRAME_NAMES = {
    mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
}
HOLD_BARS_RANGE = range(1, 11)
BARS_TOTAL = 4000
VAL_BARS = 300
TEST_BARS = 150
PROBA_BUY = 0.58
PROBA_SELL = 0.42

# КВАНТОВЫЕ ФИЛЬТРЫ V4
MIN_VAL_SHARPE = 0.85                   # Минимальный коэффициент Шарпа
MIN_VAL_RETURN = 0.01                   # Минимальный доход
MAX_CORRELATION = 0.65                  # Максимально допустимая корреляция между стратегиями

OUTPUT_DIR = "ensemble_results_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not mt5.initialize():
    print("Ошибка инициализации MT5")
    sys.exit()

def add_advanced_features(df):
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_lag_1'] = df['MACD'].shift(1)
    df['MACD_lag_2'] = df['MACD'].shift(2)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_lag_1'] = df['RSI'].shift(1)

    df['Ret_1'] = df['close'].pct_change(1)
    df['Ret_3'] = df['close'].pct_change(3)
    df['Ret_5'] = df['close'].pct_change(5)

    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['BB_pos'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df['BB_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-9)

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_rel'] = tr.rolling(14).mean() / df['close']

    features = ['MACD_lag_1', 'MACD_lag_2', 'RSI_lag_1', 'Ret_1', 'Ret_3', 'Ret_5', 'BB_pos', 'BB_width', 'ATR_rel']
    return df, features

def run_simulation_v4(df_slice, hold_bars, model, features, use_equity_filter=False):
    probs = model.predict_proba(df_slice[features])[:, 1]
    
    signals = np.zeros(len(probs))
    signals[probs >= PROBA_BUY] = 1
    signals[probs <= PROBA_SELL] = -1

    positions = np.zeros(len(signals))
    pos = 0
    hold_counter = 0

    for i in range(len(signals)):
        sig = signals[i]
        if pos == 0:
            if sig != 0:
                pos = sig
                hold_counter = 1
        else:
            hold_counter += 1
            if hold_counter > hold_bars or (sig != 0 and sig != pos):
                pos = sig if (sig != 0 and sig != pos) else 0
                hold_counter = 1 if pos != 0 else 0
        positions[i] = pos

    df_res = df_slice.copy()
    df_res['Position'] = positions
    df_res['Returns'] = df_res['close'].pct_change().fillna(0)
    df_res['PnL'] = (df_res['Position'].shift(1) * df_res['Returns']).fillna(0)
    
    if use_equity_filter:
        raw_eq = df_res['PnL'].cumsum().fillna(0)
        eq_sma = raw_eq.rolling(15, min_periods=1).mean()
        active_mask = (raw_eq >= eq_sma).astype(int).shift(1).fillna(1)
        df_res['PnL'] = df_res['PnL'] * active_mask

    df_res['Equity'] = df_res['PnL'].cumsum().fillna(0)
    
    pnl_std = df_res['PnL'].std()
    sharpe = (df_res['PnL'].mean() / (pnl_std + 1e-9)) * np.sqrt(252) if pnl_std > 0 else 0

    return df_res, sharpe, df_res['PnL'].values

def process_strategy_v4(symbol, timeframe, hold_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, BARS_TOTAL)
    if rates is None or len(rates) < BARS_TOTAL // 2:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[~df.index.dayofweek.isin([5, 6])].copy()

    df['Conv'] = gaussian_filter1d(df['close'].values, sigma=2, mode='nearest')
    df['Conv_Diff'] = df['Conv'].diff()
    df['Target_Binary'] = (df['Conv_Diff'] > 0).astype(int)

    df, features = add_advanced_features(df)
    df_clean = df.dropna().copy()

    if len(df_clean) < (VAL_BARS + TEST_BARS + 300):
        return None

    train_df = df_clean.iloc[:-(VAL_BARS + TEST_BARS)].copy()
    val_df = df_clean.iloc[-(VAL_BARS + TEST_BARS):-TEST_BARS].copy()
    test_df = df_clean.iloc[-TEST_BARS:].copy()

    model = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
    )
    model.fit(train_df[features], train_df['Target_Binary'])

    val_res, val_sharpe, val_pnl = run_simulation_v4(val_df, hold_bars, model, features, use_equity_filter=False)
    test_res, test_sharpe, test_pnl = run_simulation_v4(test_df, hold_bars, model, features, use_equity_filter=True)

    return {
        'symbol': symbol,
        'timeframe': TIMEFRAME_NAMES[timeframe],
        'hold_bars': hold_bars,
        'val_return': val_res['Equity'].iloc[-1],
        'val_sharpe': val_sharpe,
        'val_pnl': val_pnl,
        'test_return': test_res['Equity'].iloc[-1],
        'test_equity': test_res['Equity'].values,
        'test_std': np.std(test_pnl) + 1e-6
    }

# ================== ЗАПУСК ПЕРЕБОРА V4 ==================
all_results = []
print("=== Старт V4 (11 Активов | M30-H4 | Декореляция) ===")
for symbol in SYMBOLS:
    for tf in TIMEFRAMES:
        for hold in HOLD_BARS_RANGE:
            res = process_strategy_v4(symbol, tf, hold)
            if res is not None:
                all_results.append(res)

df_all = pd.DataFrame(all_results)

# 1. Фильтр качества по Валидации
passing_results = [
    r for r in all_results 
    if r['val_sharpe'] >= MIN_VAL_SHARPE and r['val_return'] >= MIN_VAL_RETURN
]

passing_results.sort(key=lambda x: x['val_sharpe'], reverse=True)

print(f"\nПрошло первичное качество: {len(passing_results)} стратегий.")

if not passing_results:
    print("Ни одна стратегия не прошла порог. Смягчите MIN_VAL_SHARPE.")
    mt5.shutdown()
    sys.exit()

# 2. ФИЛЬТР ДЕКОРРЕЛЯЦИИ (Жадный алгоритм удаления клонов)
selected_strategies = []
val_pnl_matrix = pd.DataFrame([r['val_pnl'] for r in passing_results]).T
corr_matrix = val_pnl_matrix.corr().abs()

for i, strat in enumerate(passing_results):
    too_correlated = False
    for sel in selected_strategies:
        sel_idx = passing_results.index(sel)
        if corr_matrix.iloc[i, sel_idx] > MAX_CORRELATION:
            too_correlated = True
            break
    if not too_correlated:
        selected_strategies.append(strat)

print(f"После фильтра декореляции (<{MAX_CORRELATION}) осталось: {len(selected_strategies)} НЕЗАВИСИМЫХ стратегий!")

# 3. Инверсно-Волатильное Взвешивание (Inverse Volatility Weights)
raw_weights = [1.0 / s['test_std'] for s in selected_strategies]
sum_weights = sum(raw_weights)
weights = [w / sum_weights for w in raw_weights]

for idx, s in enumerate(selected_strategies):
    s['weight'] = weights[idx]

# ================== ИТОГОВЫЙ OUT-OF-SAMPLE ТЕСТ V4 ==================
weighted_test_equities = []
for s in selected_strategies:
    weighted_test_equities.append(s['test_equity'] * s['weight'])

ensemble_v4_equity = np.sum(weighted_test_equities, axis=0)

fig, ax = plt.subplots(figsize=(14, 7))
for s in selected_strategies:
    ax.plot(s['test_equity'] * s['weight'], color='blue', alpha=0.15, linewidth=0.8)

ax.plot(ensemble_v4_equity, color='darkgreen', linewidth=2.8, label='Ensemble V4 (Decorrelated + Inverse Vol Weighting)')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title(f'VERSION 4.0 FINAL OUT-OF-SAMPLE TEST: Ensemble Return = {ensemble_v4_equity[-1]:.4f}')
ax.set_xlabel('Bar Index (Pure Test Set)')
ax.set_ylabel('Weighted Cumulative Return')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_v4_equity.png'), dpi=150)

# Экспорт финального конфига с ВЕСАМИ
df_export = pd.DataFrame([{
    'symbol': s['symbol'],
    'timeframe': s['timeframe'],
    'hold_bars': s['hold_bars'],
    'weight': s['weight']
} for s in selected_strategies])

df_export.to_csv(os.path.join(OUTPUT_DIR, 'top_strategies_v4.csv'), index=False)

print(f"\nГрафик сохранен в {OUTPUT_DIR}/ensemble_v4_equity.png")
print(f"Финальный конфиг сохранен в {OUTPUT_DIR}/top_strategies_v4.csv")

print("\n=== Отобранный независимый портфель ===")
print(df_export)

mt5.shutdown()
plt.show()