import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import optuna
import os

# --- НАСТРОЙКИ ПОИСКА ---
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY']
TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4
}
SIGMA = 2.0
LOOKAHEAD = int(4 * SIGMA)
SPREAD_COST = 0.00015 # 1.5 пункта комиссия
BARS_TOTAL = 10000
N_TRIALS_PER_COMBO = 30 # Сколько попыток на одну пару/ТФ

def get_features(df):
    d = df.copy()
    d['MACD'] = (d['close'].ewm(span=12).mean() - d['close'].ewm(span=26).mean()).shift(1)
    d['rsi'] = (pd.Series(100 - 100 / (1 + d['close'].diff().clip(lower=0).ewm(14).mean() / (d['close'].diff().clip(upper=0).abs().ewm(14).mean() + 1e-9)))).shift(1)
    d['vol'] = d['close'].rolling(20).std().shift(1)
    d['ret'] = d['close'].pct_change()
    return d.dropna()

def evaluate(symbol, tf_val, hold, sl, thr):
    rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, BARS_TOTAL)
    if rates is None: return -1
    df = get_features(pd.DataFrame(rates))
    
    # Walk-Forward (последние 2000 баров)
    test_len = 2000
    train_df = df.iloc[:-test_len].copy()
    test_df = df.iloc[-test_len:].copy()
    
    # Учитель-Ученик
    train_df['Target'] = np.where(gaussian_filter1d(train_df['close'].values, SIGMA, order=1) > 0, 1, 0)
    train_final = train_df.iloc[:-LOOKAHEAD].dropna()
    
    feats = ['MACD', 'rsi', 'vol']
    model = lgb.LGBMClassifier(n_estimators=100, verbosity=-1)
    model.fit(train_final[feats], train_final['Target'])
    
    probs = model.predict_proba(test_df[feats])[:, 1]
    
    # Симуляция (Execution Lag + Spread)
    pos = 0; h_cnt = 0; pnl = []; trades = 0
    for i in range(len(test_df)-1):
        p = probs[i]
        sig = 1 if p > thr else (-1 if p < (1-thr) else 0)
        
        if pos == 0:
            if sig != 0: 
                pos = sig; h_cnt = 0; trades += 1
        else:
            h_cnt += 1
            if h_cnt >= hold or (sig != 0 and sig != pos):
                pos = sig if (sig != 0 and sig != pos) else 0
                trades += 1 if pos != 0 else 0
        
        # Результат на следующем баре
        pnl.append(pos * test_df['ret'].iloc[i+1])
    
    total_pnl = np.sum(pnl) - (trades * SPREAD_COST)
    sharpe = (np.mean(pnl) / (np.std(pnl) + 1e-9)) * np.sqrt(252)
    return sharpe if total_pnl > 0 else -1

if not mt5.initialize(): exit()

best_configs = []

for sym in SYMBOLS:
    for tf_name, tf_val in TIMEFRAMES.items():
        print(f"Сканирую: {sym} {tf_name}...")
        
        def objective(trial):
            hold = trial.suggest_int('hold', 2, 15)
            sl = trial.suggest_int('sl', 20, 150)
            thr = trial.suggest_float('thr', 0.52, 0.65)
            return evaluate(sym, tf_val, hold, sl, thr)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS_PER_COMBO)
        
        if study.best_value > 0.5: # Берем только те, где Шарп выше 0.5
            best_configs.append({
                'symbol': sym, 'tf': tf_name, 
                'hold': study.best_params['hold'],
                'sl': study.best_params['sl'],
                'thr': study.best_params['thr'],
                'sharpe': study.best_value
            })

# Сохраняем в CSV
pd.DataFrame(best_configs).to_csv("best_configs.csv", index=False)
print("СКАНИРОВАНИЕ ЗАВЕРШЕНО. Результаты в best_configs.csv")
mt5.shutdown()