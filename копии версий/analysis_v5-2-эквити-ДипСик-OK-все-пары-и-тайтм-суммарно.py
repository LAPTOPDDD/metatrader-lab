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
HOLD_BARS_RANGE = range(1, 11)          # 1..10
BARS_TOTAL = 4000
TEST_BARS = 150
RANDOM_STATE = 42
TOP_N = 35                              # число стратегий в ансамбле
OUTPUT_DIR = "ensemble_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== ПОДКЛЮЧЕНИЕ MT5 ==================
if not mt5.initialize():
    print("Ошибка инициализации MetaTrader 5")
    sys.exit()

# ================== ФУНКЦИЯ ДЛЯ ОДНОГО ЗАПУСКА ==================
def run_backtest(symbol, timeframe, hold_bars):
    """
    Загружает данные, обучает модель, возвращает equity (массив) и метрики.
    Использует процентные изменения для сопоставимости.
    """
    timeframe_name = TIMEFRAME_NAMES.get(timeframe, str(timeframe))

    # 1. Загрузка данных
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, BARS_TOTAL)
    if rates is None or len(rates) < BARS_TOTAL // 2:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # 2. Удаляем выходные
    df = df[~df.index.dayofweek.isin([5, 6])].copy()

    # 3. Индикаторы и целевая переменная
    df['Conv'] = gaussian_filter1d(df['close'].values, sigma=2, mode='nearest')
    df['Conv_Diff'] = df['Conv'].diff()
    df['Target_Binary'] = (df['Conv_Diff'] > 0).astype(int)

    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    for i in range(1, 6):
        df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
    features = [f'MACD_lag_{i}' for i in range(1, 6)]

    df_clean = df.dropna().copy()
    if len(df_clean) < TEST_BARS + 50:
        return None

    train_df = df_clean.iloc[:-TEST_BARS].copy()
    test_df = df_clean.iloc[-TEST_BARS:].copy()

    # 4. Обучение модели
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbosity=-1
    )
    model.fit(train_df[features], train_df['Target_Binary'])
    test_df['Prediction'] = model.predict(test_df[features])

    # 5. Торговая логика
    test_df['Signal'] = (test_df['Prediction'] == 1).astype(int)
    test_df['Position'] = 0
    position = 0
    hold_counter = 0
    for i in range(len(test_df)):
        if position == 0:
            if test_df.iloc[i]['Signal'] == 1:
                position = 1
                hold_counter = 1
        else:
            hold_counter += 1
            if hold_counter > hold_bars:
                position = 0
                hold_counter = 0
        test_df.at[test_df.index[i], 'Position'] = position

    # 6. Расчёт PnL (процентные изменения)
    test_df['Returns'] = test_df['close'].pct_change().fillna(0)   # относительное изменение
    test_df['PnL'] = test_df['Position'] * test_df['Returns']
    test_df['Equity'] = test_df['PnL'].cumsum()

    total_return = test_df['Equity'].iloc[-1]
    max_drawdown = (test_df['Equity'].cummax() - test_df['Equity']).max()

    return {
        'symbol': symbol,
        'timeframe': timeframe_name,
        'hold_bars': hold_bars,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'equity': test_df['Equity'].values,   # массив длины TEST_BARS
        'dates': test_df.index                # для возможной подписи
    }

# ================== ЗАПУСК ВСЕХ КОМБИНАЦИЙ ==================
all_results = []
for symbol in SYMBOLS:
    for tf in TIMEFRAMES:
        for hold in HOLD_BARS_RANGE:
            print(f"Обработка: {symbol} {TIMEFRAME_NAMES[tf]} Hold={hold}")
            res = run_backtest(symbol, tf, hold)
            if res is not None:
                all_results.append(res)

if not all_results:
    print("Нет результатов.")
    mt5.shutdown()
    sys.exit()

# ================== ВЫБОР ТОП-N ПО ДОХОДНОСТИ ==================
df_results = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'equity' and k != 'dates'}
    for r in all_results
])
df_results_sorted = df_results.sort_values('total_return', ascending=False)
top_results = df_results_sorted.head(TOP_N)

print("\n=== Топ-35 стратегий ===")
print(top_results[['symbol', 'timeframe', 'hold_bars', 'total_return', 'max_drawdown']])

# Извлекаем equity для отобранных стратегий
selected_equities = []
for idx, row in top_results.iterrows():
    # находим соответствующий результат в all_results
    for r in all_results:
        if (r['symbol'] == row['symbol'] and 
            r['timeframe'] == row['timeframe'] and 
            r['hold_bars'] == row['hold_bars']):
            selected_equities.append(r['equity'])
            break

# ================== ПОСТРОЕНИЕ ГРАФИКА ==================
fig, ax = plt.subplots(figsize=(14, 8))

# Все equity (полупрозрачные)
for eq in selected_equities:
    ax.plot(eq, color='blue', alpha=0.15, linewidth=0.8)

# Суммарная equity
ensemble_equity = np.sum(selected_equities, axis=0)
ax.plot(ensemble_equity, color='red', linewidth=2.5, label='Ensemble (sum)')

ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_title(f'Ensemble of {TOP_N} strategies – Total Return = {ensemble_equity[-1]:.4f}')
ax.set_xlabel('Bar index (test set)')
ax.set_ylabel('Cumulative return')
ax.legend()
ax.grid(True, alpha=0.3)

# Сохраняем
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_equity.png'), dpi=150)
print(f"\nГрафик ансамбля сохранён: {os.path.join(OUTPUT_DIR, 'ensemble_equity.png')}")

# Также сохраняем список отобранных стратегий
top_results.to_csv(os.path.join(OUTPUT_DIR, 'top_35_strategies.csv'), index=False)
print(f"Список стратегий сохранён: {os.path.join(OUTPUT_DIR, 'top_35_strategies.csv')}")

mt5.shutdown()
plt.show()