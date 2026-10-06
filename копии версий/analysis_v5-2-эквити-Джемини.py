import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# Setup
symbol = "USDJPY"
if not mt5.initialize():
    quit()

# 1. Fetch (берем 3000 баров вместо 400 для нормального обучения)
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 3000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# 2. Filter weekends
df = df[~df.index.dayofweek.isin([5, 6])].copy()

# 3. Target Definition (БЕЗ утечки данных!)
# Цель: вырастет ли цена на следующей свече? (1 = Да, 0 = Нет)
df['Target_Binary'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

# Features (Каузальные индикаторы, используют только прошлые данные)
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
for i in range(1, 6):
    df[f'MACD_lag_{i}'] = df['MACD'].shift(i)

features = [f'MACD_lag_{i}' for i in range(1, 6)]

# Dropna (удаляем строки с NaN от лагов и shift(-1))
df_clean = df.dropna().copy()

# Train / Test Split
TEST_SIZE = 150
train_df = df_clean.iloc[:-TEST_SIZE]
test_df = df_clean.iloc[-TEST_SIZE:].copy()

if len(train_df) < 200:
    print("ВНИМАНИЕ: недостаточно данных для обучения ML!")

# 4. Train & Predict
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, verbosity=-1)
model.fit(train_df[features], train_df['Target_Binary'])
test_df['Prediction'] = model.predict(test_df[features])

# --- Торговая логика ---
N = 1  # Количество баров удержания позиции

test_df['Signal'] = (test_df['Prediction'] == 1).astype(int)
test_df['Position'] = 0

position = 0
hold_counter = 0

for i in range(len(test_df)):
    if position == 0:
        if test_df.iloc[i]['Signal'] == 1:
            position = 1
            hold_counter = 0
    else:
        hold_counter += 1
        # Выходим, когда продержали N баров
        if hold_counter >= N:
            position = 0
            hold_counter = 0
    
    test_df.iloc[i, test_df.columns.get_loc('Position')] = position

# PnL и Equity (Векторизовано и без Look-ahead bias)
# Position.shift(1) гарантирует, что позиция, открытая на свече t, получает доходность свечи t+1
test_df['PnL'] = test_df['Position'].shift(1) * test_df['close'].diff()
test_df['Equity'] = test_df['PnL'].cumsum().fillna(0)

# --- Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
x_idx = np.arange(len(test_df))

# Ax1: Price
ax1.plot(x_idx, test_df['close'], label='Close Price', alpha=0.7)
ax1.set_title('Close Price (Test Set)')
ax1.legend(loc='upper left')

# Ax2: Target vs Prediction
ax2.step(x_idx, test_df['Target_Binary'], label='Target (Actual)', where='post', alpha=0.5, color='blue')
ax2.step(x_idx, test_df['Prediction'], label='LGBM Prediction', where='post', linestyle='--', color='orange')
ax2.set_title('Target vs. Prediction (Test Set)')
ax2.legend(loc='upper left')

# Ax3: Equity Curve
ax3.plot(x_idx, test_df['Equity'], label='Equity Curve', color='purple', linewidth=2)
ax3.fill_between(x_idx, test_df['Equity'], alpha=0.3, color='purple')
ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax3.set_title(f'Equity Curve (Test Set, Hold={N} bar(s))')
ax3.legend(loc='upper left')

# Ось X
dates = test_df.index
step = max(1, len(dates) // 15)
tick_positions = np.arange(0, len(dates), step)
tick_labels = [dates[i].strftime('%Y-%m-%d %H:%M') for i in tick_positions]

ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=45, ha='right')

ax1.set_xticklabels([])
ax2.set_xticklabels([])

plt.tight_layout()
plt.show()

mt5.shutdown()