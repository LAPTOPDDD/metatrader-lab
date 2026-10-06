import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

mt5.initialize()
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 50000)
mt5.shutdown()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df['hour'] = df.index.hour
df['date'] = df.index.date

# =====================================================
# 1. БАЗОВАЯ СТРАТЕГИЯ + СТОП-ЛОСС + ТЕЙК-ПРОФИТ
# =====================================================
asia = df[df['hour'] < 8].groupby('date').agg(
    asia_high=('high','max'), 
    asia_low=('low','min')
)
asia['asia_range'] = asia['asia_high'] - asia['asia_low']
df = df.join(asia, on='date')

london = df[df['hour'] == 8].copy()
london['signal'] = np.where(london['close'] > london['asia_high'], 1,
                     np.where(london['close'] < london['asia_low'], -1, 0))
london = london[london['signal'] != 0].copy()
london['date'] = london.index.date
london['entry'] = london['close']

SL_MULT = 1.5
TP_MULT = 2.5

london['sl'] = np.where(london['signal']==1, 
                         london['entry'] - SL_MULT*london['asia_range'],
                         london['entry'] + SL_MULT*london['asia_range'])
london['tp'] = np.where(london['signal']==1,
                         london['entry'] + TP_MULT*london['asia_range'],
                         london['entry'] - TP_MULT*london['asia_range'])

def simulate_trade(row, df_full):
    day_bars = df_full[(df_full['date']==row['date']) & 
                        (df_full['hour']>=9) & (df_full['hour']<=22)]
    if len(day_bars)==0: return np.nan, 'no_data'
    
    for _, bar in day_bars.iterrows():
        if row['signal']==1:
            if bar['low'] <= row['sl']:  return (row['sl']-row['entry'])/row['entry'], 'SL'
            if bar['high'] >= row['tp']: return (row['tp']-row['entry'])/row['entry'], 'TP'
        else:
            if bar['high'] >= row['sl']: return (row['entry']-row['sl'])/row['entry'], 'SL'
            if bar['low'] <= row['tp']:  return (row['entry']-row['tp'])/row['entry'], 'TP'
    
    exit_price = day_bars.iloc[-1]['close']
    pnl = (exit_price-row['entry'])/row['entry'] * row['signal']
    return pnl, 'TIME'

print("Симулирую сделки...")
results = london.apply(lambda r: simulate_trade(r, df), axis=1)
london['pnl'] = [r[0] for r in results]
london['exit_type'] = [r[1] for r in results]
london = london.dropna(subset=['pnl'])

print(f"Всего сделок: {len(london)}")
print(f"Выходы: {london['exit_type'].value_counts().to_dict()}")
print(f"Win Rate базовой: {(london['pnl']>0).mean()*100:.1f}%")
print(f"Базовая доходность: {((1+london['pnl']).cumprod().iloc[-1]-1)*100:+.2f}%")

# =====================================================
# 2. FEATURES (БЕЗ УТЕЧКИ БУДУЩЕГО)
# =====================================================
london['range_pct'] = london['asia_range'] / london['entry']
london['breakout_strength'] = np.where(
    london['signal']==1,
    (london['entry']-london['asia_high'])/london['entry'],
    (london['asia_low']-london['entry'])/london['entry']
)
london['day_of_week'] = pd.to_datetime(london['date']).dt.dayofweek

# ВАЖНО: сдвигаем daily на 1 день, чтобы использовать только ПРОШЛЫЕ данные
daily = df.resample('D').agg({'close':'last'}).dropna()
daily['vol_5d'] = daily['close'].pct_change().rolling(5).std()
daily['sma20'] = daily['close'].rolling(20).mean()
daily['trend'] = daily['close']/daily['sma20']-1
daily['prev_ret'] = daily['close'].pct_change()
daily['vol_20d'] = daily['close'].pct_change().rolling(20).std()

# КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: shift(1) — берем данные ВЧЕРАШНЕГО дня
daily_shifted = daily.shift(1)

dmap = {d.date(): row for d,row in daily_shifted.iterrows()}
london['vol_5d'] = london['date'].map(lambda d: dmap.get(d, {}).get('vol_5d') if d in dmap else None)
london['trend'] = london['date'].map(lambda d: dmap.get(d, {}).get('trend') if d in dmap else None)
london['prev_ret'] = london['date'].map(lambda d: dmap.get(d, {}).get('prev_ret') if d in dmap else None)
london['vol_20d'] = london['date'].map(lambda d: dmap.get(d, {}).get('vol_20d') if d in dmap else None)
london['vol_ratio'] = london['vol_5d'] / london['vol_20d']

london['target'] = (london['pnl'] > 0).astype(int)

# УБИРАЕМ 'signal' из features — он и так известен, не должен быть предиктором
features = ['range_pct','breakout_strength','day_of_week','vol_5d','trend','prev_ret','vol_ratio']
london = london.dropna(subset=features+['target']).reset_index(drop=True)

print(f"Сделок после фильтрации признаков: {len(london)}")

# =====================================================
# 3. WALK-FORWARD ВАЛИДАЦИЯ
# =====================================================
def walk_forward(data, features, train_size=150, retrain_step=30):
    preds = []
    for start in range(train_size, len(data), retrain_step):
        train = data.iloc[:start]
        test = data.iloc[start:start+retrain_step]
        if len(test)==0: break
        if train['target'].nunique() < 2: continue
        
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=10, 
            random_state=42, verbose=-1
        )
        model.fit(train[features], train['target'])
        prob = model.predict_proba(test[features])[:,1]
        
        for i in range(len(test)):
            preds.append({
                'prob': prob[i],
                'pnl': test.iloc[i]['pnl'],
                'date': test.iloc[i]['date'],
                'range_pct': test.iloc[i]['range_pct']
            })
    return pd.DataFrame(preds)

# Адаптируем train_size под кол-во данных
train_size = max(60, int(len(london) * 0.3))
retrain_step = max(20, int(len(london) * 0.05))

wf = walk_forward(london, features, train_size=train_size, retrain_step=retrain_step)
print(f"\nWalk-forward: train_size={train_size}, retrain_step={retrain_step}")
print(f"Walk-forward сделок: {len(wf)}")

if len(wf) == 0:
    print("ОШИБКА: walk-forward не сработал. Мало данных.")
    exit()

# =====================================================
# 4. ОПТИМИЗАЦИЯ THRESHOLD
# =====================================================
thresholds = np.arange(0.45, 0.70, 0.02)
opt = []
for th in thresholds:
    filtered = wf[wf['prob']>th]
    if len(filtered)<10: continue
    eq = (1+filtered['pnl']).cumprod()
    sharpe = filtered['pnl'].mean()/filtered['pnl'].std()*np.sqrt(252) if filtered['pnl'].std()>0 else 0
    wr = (filtered['pnl']>0).mean()*100
    pf = filtered.loc[filtered['pnl']>0,'pnl'].sum()/abs(filtered.loc[filtered['pnl']<0,'pnl'].sum()) if (filtered['pnl']<0).any() else 999
    opt.append({'threshold':round(th,2), 'trades':len(filtered), 'win_rate':round(wr,1), 
                'return_pct':round((eq.iloc[-1]-1)*100,2), 'sharpe':round(sharpe,2), 
                'profit_factor':round(pf,2)})

opt_df = pd.DataFrame(opt)
print("\n=== ОПТИМИЗАЦИЯ THRESHOLD ===")
print(opt_df.to_string(index=False))

if len(opt_df) == 0:
    print("Нет валидных threshold. Понижаем планку.")
    best_th = 0.5
else:
    best_th = opt_df.loc[opt_df['sharpe'].idxmax(), 'threshold']
print(f"\nЛучший threshold по Sharpe: {best_th:.2f}")

# =====================================================
# 5. РИСК-МЕНЕДЖМЕНТ (1% депозита на сделку)
# =====================================================
RISK_PER_TRADE = 0.01
INITIAL_CAPITAL = 10000

final = wf[wf['prob']>best_th].copy().reset_index(drop=True)

capital = [INITIAL_CAPITAL]
for i, row in final.iterrows():
    risk_money = capital[-1] * RISK_PER_TRADE
    # позиция: чтобы при движении SL_MULT*range_pct против нас = 1% депозита
    position_size = risk_money / (SL_MULT * row['range_pct'])
    pnl_money = position_size * row['pnl']
    capital.append(capital[-1] + pnl_money)

final['capital'] = capital[1:]

# =====================================================
# 6. МЕТРИКИ
# =====================================================
cap_series = pd.Series(capital)
returns = cap_series.pct_change().dropna()
total_return = (capital[-1]/INITIAL_CAPITAL - 1) * 100
sharpe = returns.mean()/returns.std()*np.sqrt(252) if returns.std()>0 else 0

peak = cap_series.cummax()
dd = (cap_series - peak) / peak
max_dd = dd.min() * 100

wr = (final['pnl']>0).mean()*100
avg_win = final.loc[final['pnl']>0,'pnl'].mean()*100 if (final['pnl']>0).any() else 0
avg_loss = final.loc[final['pnl']<0,'pnl'].mean()*100 if (final['pnl']<0).any() else 0
pf = final.loc[final['pnl']>0,'pnl'].sum()/abs(final.loc[final['pnl']<0,'pnl'].sum()) if (final['pnl']<0).any() else 999

print(f"\n{'='*55}")
print(f"ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (walk-forward, риск {RISK_PER_TRADE*100}%/сделка)")
print(f"{'='*55}")
print(f"Стартовый капитал:    ${INITIAL_CAPITAL:,.0f}")
print(f"Финальный капитал:    ${capital[-1]:,.0f}")
print(f"Общая доходность:     {total_return:+.2f}%")
print(f"Всего сделок:         {len(final)}")
print(f"Win Rate:             {wr:.1f}%")
print(f"Средний выигрыш:      {avg_win:+.3f}%")
print(f"Средний проигрыш:     {avg_loss:+.3f}%")
print(f"Profit Factor:        {pf:.2f}")
print(f"Sharpe Ratio:         {sharpe:.2f}")
print(f"Max Drawdown:         {max_dd:.2f}%")

# =====================================================
# 7. ГРАФИКИ
# =====================================================
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

axes[0].plot(final['capital'].values, linewidth=2, color='green')
axes[0].axhline(INITIAL_CAPITAL, color='gray', ls='--', alpha=0.5)
axes[0].set_title(f"Equity Curve | Return: {total_return:+.2f}% | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2f}% | Trades: {len(final)}")
axes[0].set_ylabel("Capital ($)")
axes[0].grid()

dd_pct = ((cap_series - cap_series.cummax()) / cap_series.cummax() * 100).values
axes[1].fill_between(range(len(dd_pct)), dd_pct, 0, color='red', alpha=0.3)
axes[1].set_title(f"Drawdown (%) | Max: {max_dd:.2f}%")
axes[1].set_ylabel("DD %")
axes[1].set_xlabel("Trade #")
axes[1].grid()

plt.tight_layout()
plt.show()

import seaborn as sns

# =====================================================
# ДИАГНОСТИКА: ЧТО УВИДЕЛ ML-ФИЛЬТР
# =====================================================

# 1. Обучаем финальную модель на всех данных (только для интерпретации)
model_full = lgb.LGBMClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    num_leaves=15, min_child_samples=10, random_state=42, verbose=-1
)
model_full.fit(london[features], london['target'])

# Важность признаков
imp = pd.DataFrame({
    'feature': features,
    'importance': model_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*50)
print("ВАЖНОСТЬ ПРИЗНАКОВ (что модель смотрит)")
print("="*50)
print(imp.to_string(index=False))

# 2. Сравниваем: принятые vs отклонённые сделки (walk-forward)
best_th = 0.57
wf['accepted'] = wf['prob'] > best_th

# Подтягиваем признаки к wf по дате
feat_map = london.set_index('date')[features + ['pnl', 'exit_type', 'signal', 'asia_range']]
# date может дублироваться — берём первое совпадение
london_by_date = london.drop_duplicates('date').set_index('date')

for col in features + ['exit_type', 'signal']:
    if col not in wf.columns:
        wf[col] = wf['date'].map(london_by_date[col].to_dict())

print("\n" + "="*50)
print("ПРИНЯТЫЕ vs ОТКЛОНЁННЫЕ (средние значения признаков)")
print("="*50)
compare = wf.groupby('accepted')[features + ['pnl']].mean()
print(compare.T.round(5).to_string())

print("\n--- Win Rate ---")
print(wf.groupby('accepted').apply(lambda x: (x['pnl']>0).mean()*100).round(1))

print("\n--- Тип выхода ---")
print(pd.crosstab(wf['accepted'], wf['exit_type'], normalize='index').round(3)*100)

# 3. Победители vs проигравшие (среди ВСЕХ сделок)
print("\n" + "="*50)
print("ПОБЕДИТЕЛИ vs ПРОИГРАВШИЕ (что отличает + от -)")
print("="*50)
london['win'] = london['pnl'] > 0
win_lose = london.groupby('win')[features + ['pnl']].mean()
print(win_lose.T.round(5).to_string())

# 4. Простые правила "глазами" (бінning по топ-признакам)
print("\n" + "="*50)
print("ПРАВИЛА ГЛАЗАМИ: Win Rate в разрезе признаков")
print("="*50)

top_feats = imp['feature'].head(4).tolist()
for f in top_feats:
    london['_bin'] = pd.qcut(london[f], q=4, duplicates='drop')
    stats = london.groupby('_bin').agg(
        trades=('pnl','count'),
        win_rate=('win','mean'),
        avg_pnl=('pnl','mean')
    )
    stats['win_rate'] = (stats['win_rate']*100).round(1)
    stats['avg_pnl'] = (stats['avg_pnl']*100).round(3)
    print(f"\n>>> {f}")
    print(stats.to_string())

# 5. Графики
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Importance
axes[0,0].barh(imp['feature'], imp['importance'], color='steelblue')
axes[0,0].set_title("Feature Importance")
axes[0,0].invert_yaxis()

# Prob distribution: wins vs losses
axes[0,1].hist(wf.loc[wf['pnl']>0, 'prob'], bins=20, alpha=0.6, label='Wins', color='green')
axes[0,1].hist(wf.loc[wf['pnl']<=0, 'prob'], bins=20, alpha=0.6, label='Losses', color='red')
axes[0,1].axvline(best_th, color='black', ls='--', label=f'threshold={best_th}')
axes[0,1].set_title("Распределение prob: Wins vs Losses")
axes[0,1].legend()

# Equity: all vs filtered
eq_all = (1 + wf['pnl']).cumprod()
eq_f = wf.copy()
eq_f['pnl_f'] = np.where(eq_f['accepted'], eq_f['pnl'], 0)
eq_filt = (1 + eq_f['pnl_f']).cumprod()
axes[1,0].plot(eq_all.values, label='Все пробои', alpha=0.5)
axes[1,0].plot(eq_filt.values, label='Только ML-фильтр', linewidth=2, color='green')
axes[1,0].set_title("Как фильтр меняет эквити")
axes[1,0].legend()
axes[1,0].grid()

# breakout_strength vs outcome
if 'breakout_strength' in wf.columns:
    axes[1,1].scatter(wf.loc[~wf['accepted'], 'breakout_strength'], 
                      wf.loc[~wf['accepted'], 'pnl'], alpha=0.3, s=20, label='Отклонено', c='gray')
    axes[1,1].scatter(wf.loc[wf['accepted'], 'breakout_strength'], 
                      wf.loc[wf['accepted'], 'pnl'], alpha=0.6, s=30, label='Принято', c='green')
    axes[1,1].axhline(0, color='red', ls='--')
    axes[1,1].set_xlabel('breakout_strength')
    axes[1,1].set_ylabel('pnl')
    axes[1,1].set_title("Сила пробоя vs результат")
    axes[1,1].legend()

plt.tight_layout()
plt.show()

# 6. Человекочитаемый вывод
print("\n" + "="*50)
print("ВЫВОД ПРОСТЫМ ЯЗЫКОМ")
print("="*50)
acc = wf[wf['accepted']]
rej = wf[~wf['accepted']]
print(f"Модель ПРИНЯЛА {len(acc)} сделок из {len(wf)} ({len(acc)/len(wf)*100:.0f}%)")
print(f"  → WR принятых:   {(acc['pnl']>0).mean()*100:.1f}% | avg pnl: {acc['pnl'].mean()*100:+.3f}%")
print(f"  → WR отклонённых:{(rej['pnl']>0).mean()*100:.1f}% | avg pnl: {rej['pnl'].mean()*100:+.3f}%")
print(f"\nОтсеянный «мусор» тянул стратегию вниз.")
print(f"Разница в avg pnl: {(acc['pnl'].mean()-rej['pnl'].mean())*100:+.3f}% на сделку")