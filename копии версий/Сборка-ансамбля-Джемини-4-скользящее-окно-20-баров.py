import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import os

# ================== НАСТРОЙКИ WALK-FORWARD ==================
SYMBOLS = [
    "USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "XAUUSD"
]
TIMEFRAMES = [
    mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4
]
TIMEFRAME_NAMES = {
    mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4"
}
HOLD_BARS_RANGE = range(1, 11)
BARS_TOTAL = 4000

# ПАРАМЕТРЫ SKOL'ZASHCHEGO OKNA
N_STEPS = 10                            # 10 шагов переобучения
STEP_BARS = 20                          # Каждые 20 баров — переобучение!
TOTAL_OOS_BARS = N_STEPS * STEP_BARS   # Всего 200 баров на тесте

PROBA_BUY = 0.58
PROBA_SELL = 0.42
MIN_VAL_SHARPE = 0.85
MAX_CORRELATION = 0.65

OUTPUT_DIR = "walk_forward_results"
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

def run_walk_forward_for_strategy(symbol, timeframe, hold_bars):
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

    if len(df_clean) < (TOTAL_OOS_BARS + 500):
        return None

    # Цепочка результатов Walk-Forward
    oos_pnl_chain = []

    # ПРОГОН ПО 10 ШАГАМ С ПЕРЕОБУЧЕНИЕМ
    for s in range(N_STEPS):
        test_start_idx = len(df_clean) - (N_STEPS - s) * STEP_BARS
        test_end_idx = test_start_idx + STEP_BARS
        
        # Обучающая выборка росла и заканчивается ПРЯМО ПЕРЕД текущими 20 барами!
        train_df = df_clean.iloc[:test_start_idx].copy()
        # Тестовый отрезок (20 баров + 1 бар сзади для бесшовного shift(1))
        test_df = df_clean.iloc[test_start_idx - 1 : test_end_idx].copy()

        model = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
        )
        model.fit(train_df[features], train_df['Target_Binary'])

        # Прогноз на 20 баров
        probs = model.predict_proba(test_df[features])[:, 1]
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

        test_df['Position'] = positions
        test_df['Returns'] = test_df['close'].pct_change().fillna(0)
        test_df['PnL'] = (test_df['Position'].shift(1) * test_df['Returns']).fillna(0)

        # Отрезаем служебный 1-й бар и сохраняем чистые 20 баров
        step_pnl = test_df['PnL'].iloc[1:].values
        oos_pnl_chain.extend(step_pnl)

    full_oos_pnl = np.array(oos_pnl_chain)
    full_oos_equity = np.cumsum(full_oos_pnl)
    
    pnl_std = np.std(full_oos_pnl)
    sharpe = (np.mean(full_oos_pnl) / (pnl_std + 1e-9)) * np.sqrt(252) if pnl_std > 0 else 0

    return {
        'symbol': symbol,
        'timeframe': TIMEFRAME_NAMES[timeframe],
        'hold_bars': hold_bars,
        'wf_return': full_oos_equity[-1],
        'wf_sharpe': sharpe,
        'wf_equity': full_oos_equity,
        'wf_pnl': full_oos_pnl,
        'std': pnl_std + 1e-6
    }

# ================== ЗАПУСК WALK-FORWARD ДЛЯ ВСЕХ КАНДИДАТОВ ==================
print("=== Старт Walk-Forward Анализа (10 переобучений по 20 баров) ===")
results = []
for symbol in SYMBOLS:
    for tf in TIMEFRAMES:
        for hold in HOLD_BARS_RANGE:
            res = run_walk_forward_for_strategy(symbol, tf, hold)
            if res is not None:
                results.append(res)
                print(f"WF Готово: {symbol} {TIMEFRAME_NAMES[tf]} Hold={hold} | Return: {res['wf_return']:.4f} | Sharpe: {res['wf_sharpe']:.2f}")

# ФИЛЬТРАЦИЯ КАЧЕСТВА И ДЕКОРРЕЛЯЦИЯ
passing = [r for r in results if r['wf_sharpe'] >= MIN_VAL_SHARPE and r['wf_return'] > 0]
passing.sort(key=lambda x: x['wf_sharpe'], reverse=True)

if not passing:
    print("\nНи одна стратегия не прошла фильтр Sharpe >= 0.85. Используем лучшие по Return...")
    passing = sorted(results, key=lambda x: x['wf_return'], reverse=True)[:15]

# Декореляция
selected = []
pnl_matrix = pd.DataFrame([r['wf_pnl'] for r in passing]).T
corr_matrix = pnl_matrix.corr().abs()

for i, strat in enumerate(passing):
    too_corr = False
    for sel in selected:
        sel_idx = passing.index(sel)
        if corr_matrix.iloc[i, sel_idx] > MAX_CORRELATION:
            too_corr = True
            break
    if not too_corr:
        selected.append(strat)

# Взвешивание
raw_w = [1.0 / s['std'] for s in selected]
sum_w = sum(raw_w)
weights = [w / sum_w for w in raw_w]

# ПОСТРОЕНИЕ ИТОГОВОГО WALK-FORWARD ГРАФИКА
wf_ensemble_equity = np.zeros(TOTAL_OOS_BARS)
for idx, s in enumerate(selected):
    wf_ensemble_equity += s['wf_equity'] * weights[idx]

fig, ax = plt.subplots(figsize=(14, 7))
for idx, s in enumerate(selected):
    ax.plot(s['wf_equity'] * weights[idx], color='blue', alpha=0.15, linewidth=0.8)

ax.plot(wf_ensemble_equity, color='crimson', linewidth=3.0, label='Walk-Forward Ensemble (10 Retrainings)')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title(f'TRUE WALK-FORWARD EQUITY (10 Retrainings x 20 Bars) | Total Return = {wf_ensemble_equity[-1]:.4f}')
ax.set_xlabel('Bar Index (Concatenated 10x20 Out-Of-Sample Windows)')
ax.set_ylabel('Weighted Return')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'walk_forward_ensemble.png'), dpi=150)

df_export = pd.DataFrame([{
    'symbol': s['symbol'],
    'timeframe': s['timeframe'],
    'hold_bars': s['hold_bars'],
    'weight': weights[idx]
} for idx, s in enumerate(selected)])

df_export.to_csv(os.path.join(OUTPUT_DIR, 'walk_forward_strategies.csv'), index=False)

print(f"\n=== WALK-FORWARD УСПЕШНО ЗАВЕРШЕН ===")
print(f"График сохранен в {OUTPUT_DIR}/walk_forward_ensemble.png")
print(df_export)

mt5.shutdown()
plt.show()