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
HOLD_BARS_RANGE = range(1, 11)  # 1..10
BARS_TOTAL = 4000
TEST_BARS = 150
RANDOM_STATE = 42

# Папка для сохранения графиков (создаём, если нет)
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== ПОДКЛЮЧЕНИЕ MT5 ==================
if not mt5.initialize():
    print("Ошибка инициализации MetaTrader 5")
    sys.exit()

# ================== ФУНКЦИЯ ДЛЯ ОДНОГО ЗАПУСКА ==================
def run_backtest(symbol, timeframe, hold_bars, show_plot=False, save_plot=True):
    """
    Загружает данные, обучает модель, строит графики и сохраняет/показывает.
    Возвращает словарь с метриками.
    """
    timeframe_name = TIMEFRAME_NAMES.get(timeframe, str(timeframe))
    print(f"\n--- Обработка: {symbol} {timeframe_name} Hold={hold_bars} ---")

    # 1. Загрузка данных
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, BARS_TOTAL)
    if rates is None or len(rates) < BARS_TOTAL // 2:  # если данных мало
        print(f"  Пропуск: недостаточно данных для {symbol} {timeframe_name}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # 2. Удаляем выходные (суббота=5, воскресенье=6)
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
        print(f"  Пропуск: слишком мало данных после очистки ({len(df_clean)})")
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

    # 6. Расчёт PnL и метрик
    test_df['Returns'] = test_df['close'].diff().fillna(0)
    test_df['PnL'] = test_df['Position'] * test_df['Returns']
    test_df['Equity'] = test_df['PnL'].cumsum()

    total_return = test_df['Equity'].iloc[-1]
    max_drawdown = (test_df['Equity'].cummax() - test_df['Equity']).max()
    print(f"  Доходность: {total_return:.4f}, Просадка: {max_drawdown:.4f}")

    # 7. Построение графика
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    x_idx = np.arange(len(test_df))

    ax1.plot(x_idx, test_df['close'], label='Close', alpha=0.5)
    ax1.plot(x_idx, test_df['Conv'], label='Conv', color='red')
    ax1.set_title(f'{symbol} {timeframe_name}  Hold={hold_bars}  |  Return={total_return:.3f}')
    ax1.legend(loc='upper left')

    ax2.plot(x_idx, test_df['Conv_Diff'], label='Conv Diff', color='green', alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.legend(loc='upper left')

    ax3.step(x_idx, test_df['Target_Binary'], label='Target', where='post', alpha=0.5, color='blue')
    ax3.step(x_idx, test_df['Prediction'], label='Prediction', where='post', linestyle='--', color='orange')
    ax3.legend(loc='upper left')

    ax4.plot(x_idx, test_df['Equity'], label='Equity', color='purple', linewidth=2)
    ax4.fill_between(x_idx, test_df['Equity'], alpha=0.3, color='purple')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_title(f'Equity (Hold={hold_bars})')
    ax4.legend(loc='upper left')

    # Оформление оси времени
    dates = test_df.index
    step = max(1, len(dates) // 15)
    tick_positions = np.arange(0, len(dates), step)
    tick_labels = [dates[i].strftime('%Y-%m-%d %H:%M') for i in tick_positions]
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax3.tick_params(labelbottom=False)

    plt.tight_layout()

    # Сохранение
    if save_plot:
        filename = f"{symbol}_{timeframe_name}_Hold{hold_bars}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  График сохранён: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        'symbol': symbol,
        'timeframe': timeframe_name,
        'hold_bars': hold_bars,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'n_train': len(train_df),
        'n_test': len(test_df),
    }

# ================== ОСНОВНОЙ ЦИКЛ ==================
results = []
for symbol in SYMBOLS:
    for tf in TIMEFRAMES:
        for hold in HOLD_BARS_RANGE:
            stats = run_backtest(symbol, tf, hold, show_plot=False, save_plot=True)
            if stats is not None:
                results.append(stats)

# Сохраняем сводную таблицу метрик
if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print("\n=== Сводка по всем комбинациям ===")
    print(df_results.to_string())
else:
    print("Ни одна комбинация не была обработана.")

mt5.shutdown()
print("Готово.")