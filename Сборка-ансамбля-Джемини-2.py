import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import os

# ================== НАСТРОЙКИ V2 ==================
SYMBOLS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD"]
TIMEFRAMES = [
    mt5.TIMEFRAME_M5,
    mt5.TIMEFRAME_M15,
    mt5.TIMEFRAME_M30,
    mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_H4,
]
TIMEFRAME_NAMES = {
    mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M15: "M15",
    mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1",
    mt5.TIMEFRAME_H4: "H4",
}
HOLD_BARS_RANGE = range(1, 11)
BARS_TOTAL = 4000
VAL_BARS = 300                          # Для выбора Топ-35
TEST_BARS = 150                         # Чистый Out-of-Sample тест
PROBA_BUY = 0.58                        # Порог уверенности для BUY (58%)
PROBA_SELL = 0.42                       # Порог уверенности для SELL (58%)
TOP_N = 35
OUTPUT_DIR = "ensemble_results_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not mt5.initialize():
    print("Ошибка инициализации MT5")
    sys.exit()

def add_advanced_features(df):
    """Генерация комплексного набора каузальных признаков"""
    # 1. MACD
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_lag_1'] = df['MACD'].shift(1)
    df['MACD_lag_2'] = df['MACD'].shift(2)

    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_lag_1'] = df['RSI'].shift(1)

    # 3. Momentum (Доходности)
    df['Ret_1'] = df['close'].pct_change(1)
    df['Ret_3'] = df['close'].pct_change(3)
    df['Ret_5'] = df['close'].pct_change(5)

    # 4. Bollinger Bands
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['BB_pos'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df['BB_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-9)

    # 5. Relative ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_rel'] = tr.rolling(14).mean() / df['close']

    features = ['MACD_lag_1', 'MACD_lag_2', 'RSI_lag_1', 'Ret_1', 'Ret_3', 'Ret_5', 'BB_pos', 'BB_width', 'ATR_rel']
    return df, features

def run_simulation_v2(df_slice, hold_bars, model, features):
    """Симуляция торговли с учетом вероятностей и фильтра уверенности"""
    probs = model.predict_proba(df_slice[features])[:, 1]
    
    # Генерация сигналов с порогом уверенности: 1 (BUY), -1 (SELL), 0 (CASH)
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
            # Выход по таймауту ИЛИ по смене уверенного сигнала
            if hold_counter > hold_bars or (sig != 0 and sig != pos):
                pos = sig if (sig != 0 and sig != pos) else 0
                hold_counter = 1 if pos != 0 else 0
        positions[i] = pos

    df_res = df_slice.copy()
    df_res['Position'] = positions
    df_res['Returns'] = df_res['close'].pct_change().fillna(0)
    # Честный PnL с упреждением
    df_res['PnL'] = df_res['Position'].shift(1) * df_res['Returns']
    df_res['Equity'] = df_res['PnL'].cumsum().fillna(0)
    return df_res

def process_strategy_v2(symbol, timeframe, hold_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, BARS_TOTAL)
    if rates is None or len(rates) < BARS_TOTAL // 2:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[~df.index.dayofweek.isin([5, 6])].copy()

    # Target (Идеальный разметочник)
    df['Conv'] = gaussian_filter1d(df['close'].values, sigma=2, mode='nearest')
    df['Conv_Diff'] = df['Conv'].diff()
    df['Target_Binary'] = (df['Conv_Diff'] > 0).astype(int)

    # Генерация признаков
    df, features = add_advanced_features(df)
    df_clean = df.dropna().copy()

    if len(df_clean) < (VAL_BARS + TEST_BARS + 300):
        return None

    train_df = df_clean.iloc[:-(VAL_BARS + TEST_BARS)].copy()
    val_df = df_clean.iloc[-(VAL_BARS + TEST_BARS):-TEST_BARS].copy()
    test_df = df_clean.iloc[-TEST_BARS:].copy()

    # Обучение модели
    model = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1
    )
    model.fit(train_df[features], train_df['Target_Binary'])

    # Симуляции
    val_res = run_simulation_v2(val_df, hold_bars, model, features)
    test_res = run_simulation_v2(test_df, hold_bars, model, features)

    return {
        'symbol': symbol,
        'timeframe': TIMEFRAME_NAMES[timeframe],
        'hold_bars': hold_bars,
        'val_return': val_res['Equity'].iloc[-1],
        'test_return': test_res['Equity'].iloc[-1],
        'test_equity': test_res['Equity'].values,
    }

# ================== ЗАПУСК ГРИД-СЕРЧА V2 ==================
all_results = []
print("=== Старт тестирования V2 (250 комбинаций с фильтром уверенности) ===")
for symbol in SYMBOLS:
    for tf in TIMEFRAMES:
        for hold in HOLD_BARS_RANGE:
            res = process_strategy_v2(symbol, tf, hold)
            if res is not None:
                all_results.append(res)
                print(f"Готово: {symbol} {TIMEFRAME_NAMES[tf]} Hold={hold} | Val Return: {res['val_return']:.4f}")

df_all = pd.DataFrame(all_results)
df_sorted = df_all.sort_values('val_return', ascending=False)
top_strategies = df_sorted.head(TOP_N).copy()

print(f"\n=== ТОП-{TOP_N} СТРАТЕГИЙ V2 (Отбор по Validation set) ===")
print(top_strategies[['symbol', 'timeframe', 'hold_bars', 'val_return', 'test_return']])

# ================== ЧИСТЫЙ OUT-OF-SAMPLE ТЕСТ V2 ==================
selected_test_equities = top_strategies['test_equity'].tolist()
ensemble_test_equity = np.sum(selected_test_equities, axis=0)

fig, ax = plt.subplots(figsize=(14, 7))
for eq in selected_test_equities:
    ax.plot(eq, color='blue', alpha=0.12, linewidth=0.8)

ax.plot(ensemble_test_equity, color='green', linewidth=2.5, label='Ensemble V2 (Pure Out-Of-Sample Test)')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title(f'VERSION 2.0 OUT-OF-SAMPLE TEST: Ensemble Return = {ensemble_test_equity[-1]:.4f}')
ax.set_xlabel('Bar Index (Pure Test Set)')
ax.set_ylabel('Cumulative Return')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_v2_equity.png'), dpi=150)
top_strategies[['symbol', 'timeframe', 'hold_bars']].to_csv(
    os.path.join(OUTPUT_DIR, 'top_35_strategies_v2.csv'), index=False
)

print(f"\nГрафик сохранен в {OUTPUT_DIR}/ensemble_v2_equity.png")
print(f"Конфигурация сохранена в {OUTPUT_DIR}/top_35_strategies_v2.csv")

mt5.shutdown()
plt.show()