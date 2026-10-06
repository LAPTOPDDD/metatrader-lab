import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import os

# ================== НАСТРОЙКИ ==================
SYMBOLS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD"]
TIMEFRAMES = [
    mt5.TIMEFRAME_M5,
    mt5.TIMEFRAME_M15,
    mt5.TIMEFRAME_M30,
    mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_H4,
]
TIMEFRAME_NAMES = {
    mt5.TIMEFRAME_M5: "M5",
    mt5.TIMEFRAME_M15: "M15",
    mt5.TIMEFRAME_M30: "M30",
    mt5.TIMEFRAME_H1: "H1",
    mt5.TIMEFRAME_H4: "H4",
}
HOLD_BARS_RANGE = range(1, 11)          # 1..10 баров
BARS_TOTAL = 4000                       # Всего баров
VAL_BARS = 300                          # Длина валидации (для выбора Топ-35)
TEST_BARS = 150                         # Длина чистого теста (Out-of-Sample)
RANDOM_STATE = 42
TOP_N = 35                              # Стратегий в ансамбле
OUTPUT_DIR = "ensemble_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== ПОДКЛЮЧЕНИЕ MT5 ==================
if not mt5.initialize():
    print("Ошибка инициализации MetaTrader 5")
    sys.exit()

def run_simulation(df_slice, hold_bars):
    """Вспомогательная функция расчета позиций и PnL с верным сдвигом shift(1)"""
    signals = (df_slice['Prediction'] == 1).astype(int).values
    positions = np.zeros(len(signals))
    
    pos = 0
    hold_counter = 0
    for i in range(len(signals)):
        if pos == 0:
            if signals[i] == 1:
                pos = 1
                hold_counter = 1
        else:
            hold_counter += 1
            if hold_counter > hold_bars:
                pos = 0
                hold_counter = 0
        positions[i] = pos

    df_res = df_slice.copy()
    df_res['Position'] = positions
    df_res['Returns'] = df_res['close'].pct_change().fillna(0)
    # КРИТИЧЕСКИ ВАЖНО: shift(1) убирает заглядывание в будущее
    df_res['PnL'] = df_res['Position'].shift(1) * df_res['Returns']
    df_res['Equity'] = df_res['PnL'].cumsum().fillna(0)
    return df_res

def process_strategy(symbol, timeframe, hold_bars):
    timeframe_name = TIMEFRAME_NAMES.get(timeframe, str(timeframe))

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, BARS_TOTAL)
    if rates is None or len(rates) < BARS_TOTAL // 2:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[~df.index.dayofweek.isin([5, 6])].copy()

    # Индикаторы и Таргет
    df['Conv'] = gaussian_filter1d(df['close'].values, sigma=2, mode='nearest')
    df['Conv_Diff'] = df['Conv'].diff()
    df['Target_Binary'] = (df['Conv_Diff'] > 0).astype(int)

    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    for i in range(1, 6):
        df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
    features = [f'MACD_lag_{i}' for i in range(1, 6)]

    df_clean = df.dropna().copy()
    if len(df_clean) < (VAL_BARS + TEST_BARS + 200):
        return None

    # 3-Этапное разделение
    train_df = df_clean.iloc[:-(VAL_BARS + TEST_BARS)].copy()
    val_df = df_clean.iloc[-(VAL_BARS + TEST_BARS):-TEST_BARS].copy()
    test_df = df_clean.iloc[-TEST_BARS:].copy()

    # Обучение модели
    model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.03, max_depth=3,
        random_state=RANDOM_STATE, verbosity=-1
    )
    model.fit(train_df[features], train_df['Target_Binary'])

    # Прогнозы
    val_df['Prediction'] = model.predict(val_df[features])
    test_df['Prediction'] = model.predict(test_df[features])

    # Симуляция на Валидации (для отбора) и на Тесте (для финальной проверки)
    val_res = run_simulation(val_df, hold_bars)
    test_res = run_simulation(test_df, hold_bars)

    val_return = val_res['Equity'].iloc[-1]
    test_return = test_res['Equity'].iloc[-1]

    return {
        'symbol': symbol,
        'timeframe': timeframe_name,
        'hold_bars': hold_bars,
        'val_return': val_return,
        'test_return': test_return,
        'test_equity': test_res['Equity'].values,
    }

# ================== ЗАПУСК ПЕРЕБОРА (250 КОМБИНАЦИЙ) ==================
all_results = []
print("=== Старт тестирования 250 комбинаций ===")
for symbol in SYMBOLS:
    for tf in TIMEFRAMES:
        for hold in HOLD_BARS_RANGE:
            res = process_strategy(symbol, tf, hold)
            if res is not None:
                all_results.append(res)
                print(f"Готово: {symbol} {TIMEFRAME_NAMES[tf]} Hold={hold} | Val Return: {res['val_return']:.4f}")

df_all = pd.DataFrame(all_results)

# ================== ЧЕСТНЫЙ ОТБОР ТОП-N ПО ВАЛИДАЦИИ ==================
df_sorted = df_all.sort_values('val_return', ascending=False)
top_strategies = df_sorted.head(TOP_N).copy()

print(f"\n=== ТОП-{TOP_N} СТРАТЕГИЙ (Отбор по Validation set) ===")
print(top_strategies[['symbol', 'timeframe', 'hold_bars', 'val_return', 'test_return']])

# ================== ОЦЕНКА АНСАМБЛЯ НА ЧИСТОМ TEST SET ==================
selected_test_equities = top_strategies['test_equity'].tolist()
ensemble_test_equity = np.sum(selected_test_equities, axis=0)

# Построение графика только Out-of-Sample участка!
fig, ax = plt.subplots(figsize=(14, 7))

for eq in selected_test_equities:
    ax.plot(eq, color='blue', alpha=0.12, linewidth=0.8)

ax.plot(ensemble_test_equity, color='red', linewidth=2.5, label='Ensemble (Sum of Top-35 on Unseen Test)')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title(f'PURE OUT-OF-SAMPLE TEST: Ensemble Return = {ensemble_test_equity[-1]:.4f}')
ax.set_xlabel('Bar Index (Pure Test Set)')
ax.set_ylabel('Cumulative Return')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_equity.png'), dpi=150)
print(f"\nГрафик сохранен в {OUTPUT_DIR}/ensemble_equity.png")

# Сохранение CSV для бота
top_strategies[['symbol', 'timeframe', 'hold_bars']].to_csv(
    os.path.join(OUTPUT_DIR, 'top_35_strategies.csv'), index=False
)
print(f"Файл конфигурации сохранен в {OUTPUT_DIR}/top_35_strategies.csv")

mt5.shutdown()
plt.show()