import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys

# ------------------- НАСТРОЙКИ -------------------
SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_H4
BARS_TOTAL = 400
TEST_BARS = 150
HOLD_BARS = 2          # удерживаем позицию ровно HOLD_BARS баров
RANDOM_STATE = 42

# ------------------- ПОДКЛЮЧЕНИЕ -------------------
if not mt5.initialize():
    print("Ошибка инициализации MetaTrader 5")
    sys.exit()

# 1. Загрузка данных
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_TOTAL)
if rates is None or len(rates) < BARS_TOTAL:
    print("Недостаточно данных от MetaTrader")
    mt5.shutdown()
    sys.exit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# 2. Удаляем выходные (суббота=5, воскресенье=6) – для H4 это корректно
df = df[~df.index.dayofweek.isin([5, 6])].copy()

# 3. Индикаторы и целевая переменная
df['Conv'] = gaussian_filter1d(df['close'].values, sigma=2, mode='nearest')
df['Conv_Diff'] = df['Conv'].diff()

# Целевая переменная: 1 – рост Conv, 0 – падение (бинарная классификация)
df['Target_Binary'] = (df['Conv_Diff'] > 0).astype(int)

# Признаки: лаги MACD
df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)
features = [f'MACD_lag_{i}' for i in range(1, 6)]

# Удаляем строки с NaN
df_clean = df.dropna().copy()
print(f"Всего строк после очистки: {len(df_clean)}")

# Проверка: хватает ли данных для обучения и теста
if len(df_clean) < TEST_BARS + 50:
    print("Слишком мало данных после фильтрации и dropna. Завершение.")
    mt5.shutdown()
    sys.exit()

# 4. Разделение на обучающую и тестовую выборки (последние TEST_BARS – тест)
train_df = df_clean.iloc[:-TEST_BARS].copy()
test_df = df_clean.iloc[-TEST_BARS:].copy()

print(f"Обучающая выборка: {len(train_df)} баров")
print(f"Тестовая выборка: {len(test_df)} баров")

# 5. Обучение модели LightGBM
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    verbosity=-1
)
model.fit(train_df[features], train_df['Target_Binary'])

# Прогноз на тесте
test_df['Prediction'] = model.predict(test_df[features])

# 6. Торговая логика (вход по сигналу Prediction == 1, удержание HOLD_BARS баров)
test_df['Signal'] = (test_df['Prediction'] == 1).astype(int)
test_df['Position'] = 0

position = 0
hold_counter = 0

for i in range(len(test_df)):
    if position == 0:
        if test_df.iloc[i]['Signal'] == 1:
            position = 1
            hold_counter = 1          # первый бар удержания (текущий)
    else:
        hold_counter += 1             # увеличиваем счётчик каждый следующий бар
        if hold_counter > HOLD_BARS:  # если удержали больше заданного числа
            position = 0
            hold_counter = 0
    test_df.at[test_df.index[i], 'Position'] = position

# 7. Расчёт PnL и Equity (с коррекцией первого diff)
test_df['Returns'] = test_df['close'].diff().fillna(0)
test_df['PnL'] = test_df['Position'] * test_df['Returns']
test_df['Equity'] = test_df['PnL'].cumsum()

# Дополнительная статистика
total_return = test_df['Equity'].iloc[-1]
max_drawdown = (test_df['Equity'].cummax() - test_df['Equity']).max()
print(f"Итоговая доходность: {total_return:.4f}")
print(f"Максимальная просадка: {max_drawdown:.4f}")

# 8. Построение графиков (только тестовый участок)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
x_idx = np.arange(len(test_df))

# График цены и свёртки
ax1.plot(x_idx, test_df['close'], label='Close Price', alpha=0.5)
ax1.plot(x_idx, test_df['Conv'], label='Gaussian Conv', color='red')
ax1.set_title('Close Price & Gaussian Convolution (Test Set)')
ax1.legend(loc='upper left')

# Производная свёртки
ax2.plot(x_idx, test_df['Conv_Diff'], label='Conv Diff', color='green', alpha=0.7)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title('First Derivative of Convolution')
ax2.legend(loc='upper left')

# Целевая переменная vs предсказания
ax3.step(x_idx, test_df['Target_Binary'], label='Target (0/1)', where='post', alpha=0.5, color='blue')
ax3.step(x_idx, test_df['Prediction'], label='LGBM Prediction', where='post', linestyle='--', color='orange')
ax3.set_title('Target vs Prediction')
ax3.legend(loc='upper left')

# Equity кривая
ax4.plot(x_idx, test_df['Equity'], label='Equity Curve', color='purple', linewidth=2)
ax4.fill_between(x_idx, test_df['Equity'], alpha=0.3, color='purple')
ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax4.set_title(f'Equity Curve (Hold={HOLD_BARS} bar(s))')
ax4.legend(loc='upper left')

# Оформление оси времени
dates = test_df.index
step = max(1, len(dates) // 15)
tick_positions = np.arange(0, len(dates), step)
tick_labels = [dates[i].strftime('%Y-%m-%d %H:%M') for i in tick_positions]
ax4.set_xticks(tick_positions)
ax4.set_xticklabels(tick_labels, rotation=45, ha='right')

# Убираем подписи на верхних графиках
ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)
ax3.tick_params(labelbottom=False)

plt.tight_layout()
plt.show()

# 9. Завершение работы с MetaTrader
mt5.shutdown()