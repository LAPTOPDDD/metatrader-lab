import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import classification_report

mt5.initialize()
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 15000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df['hour'] = df.index.hour
df['date'] = df.index.date

# --- 1. Базовая стратегия London Breakout ---
asia = df[df['hour'] < 8].groupby('date').agg(
    asia_high=('high','max'), 
    asia_low=('low','min'),
    asia_range=('high', lambda x: x.max()-x.min())
)
df = df.join(asia, on='date')

london = df[df['hour'] == 8].copy()
london['signal'] = np.where(london['close'] > london['asia_high'], 1,
                     np.where(london['close'] < london['asia_low'], -1, 0))

close_22 = df[df['hour'] == 22].reset_index()
close_22['date'] = close_22['time'].dt.date
close_22 = close_22.set_index('date')['close']

london['date'] = london.index.date
london['entry'] = london['close']
london['exit'] = london['date'].map(close_22)
london['pnl'] = (london['exit'] - london['entry']) / london['entry'] * london['signal']
london = london.dropna(subset=['pnl'])
london = london[london['signal'] != 0].copy()

# --- 2. FEATURES для мета-модели ---
# Признаки, доступные В МОМЕНТ входа (08:00)
london['range_pct'] = london['asia_range'] / london['entry']  # ширина азиатского диапазона
london['breakout_strength'] = np.where(
    london['signal'] == 1,
    (london['entry'] - london['asia_high']) / london['entry'],
    (london['asia_low'] - london['entry']) / london['entry']
)
london['day_of_week'] = pd.to_datetime(london['date']).dt.dayofweek

# Волатильность за последние 5 дней (по дневным свечам)
daily = df.resample('D').agg({'close':'last'}).dropna()
daily['vol_5d'] = daily['close'].pct_change().rolling(5).std()
london['vol_5d'] = london['date'].map(daily['vol_5d'].to_dict() if False else 
                                       {d.date(): v for d,v in daily['vol_5d'].items()})

# Тренд: где мы относительно 20-дневной средней
daily['sma20'] = daily['close'].rolling(20).mean()
daily['trend'] = daily['close'] / daily['sma20'] - 1
london['trend'] = london['date'].map({d.date(): v for d,v in daily['trend'].items()})

# Импульс предыдущего дня
daily['prev_ret'] = daily['close'].pct_change()
london['prev_ret'] = london['date'].map({d.date(): v for d,v in daily['prev_ret'].items()})

# Целевая переменная: успешный трейд = 1
london['target'] = (london['pnl'] > 0).astype(int)

features = ['range_pct', 'breakout_strength', 'day_of_week', 'vol_5d', 'trend', 'prev_ret', 'signal']
london = london.dropna(subset=features + ['target'])

# --- 3. Out-of-Sample split ---
split = int(len(london) * 0.6)
train = london.iloc[:split]
test = london.iloc[split:].copy()

X_train, y_train = train[features], train['target']
X_test, y_test = test[features], test['target']

# --- 4. LightGBM ---
model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

test['prob'] = model.predict_proba(X_test)[:, 1]

# --- 5. Сравнение эквити ---
threshold = 0.55  # порог уверенности

test['pnl_base'] = test['pnl']  # без фильтра
test['pnl_ml'] = np.where(test['prob'] > threshold, test['pnl'], 0)

eq_base = (1 + test['pnl_base']).cumprod()
eq_ml = (1 + test['pnl_ml']).cumprod()

n_base = len(test)
n_ml = (test['prob'] > threshold).sum()
wr_base = (test['pnl_base'] > 0).mean() * 100
wr_ml = (test.loc[test['prob'] > threshold, 'pnl'] > 0).mean() * 100

print(f"=== БЕЗ ФИЛЬТРА ===")
print(f"Сделок: {n_base} | Win Rate: {wr_base:.1f}% | Итог: {(eq_base.iloc[-1]-1)*100:+.2f}%")
print(f"=== С LIGHTGBM ФИЛЬТРОМ (prob > {threshold}) ===")
print(f"Сделок: {n_ml} | Win Rate: {wr_ml:.1f}% | Итог: {(eq_ml.iloc[-1]-1)*100:+.2f}%")

# --- 6. График ---
plt.figure(figsize=(13,6))
plt.plot(eq_base.values, label=f'London Breakout (базовая): {(eq_base.iloc[-1]-1)*100:+.2f}%', alpha=0.6)
plt.plot(eq_ml.values, label=f'+ LightGBM фильтр: {(eq_ml.iloc[-1]-1)*100:+.2f}%', linewidth=2, color='red')
plt.title("London Breakout EURUSD H1: Base vs Meta-Labeling (LightGBM)")
plt.xlabel("Сделки (Out-of-Sample)")
plt.ylabel("Equity")
plt.legend()
plt.grid()
plt.show()

# Важность признаков
importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print("\n=== Важность признаков ===")
print(importance)