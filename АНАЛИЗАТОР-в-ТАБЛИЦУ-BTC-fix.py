import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import optuna

# --- НАСТРОЙКИ ---
SYMBOLS = [
    'AAVE', 'BITCOIN', 'BITCOIN CASH', 'CARDANO', 'CHAINLINK', 'DOGECOIN', 
    'ETHEREUM', 'LITECOIN', 'POLKADOT', 'POLYGON', 'RIPPLE', 'SOLANA', 
    'STELLAR', 'TRON', 'UNISWAP', 'AVALANCH', 'AXS', 'BNB', 'COSMOS', 
    'PYTH', 'ILLUVIUM', 'INJ', 'RENDER', 'TRUMP', 'SUI', 'HEDERA', 
    'NEAR', 'APTOS', 'ARBITRUM', 'TONCOIN', 'PAXG'
]

TIMEFRAMES = {"M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1}
SIGMA = 2.0
LOOKAHEAD = 8
SPREAD_COST = 0.00015
BARS_TOTAL = 8000 
N_TRIALS_PER_COMBO = 20 # Попыток на каждую пару

def get_features(df):
    """СИНХРОНИЗИРОВАНО С БОТОМ"""
    d = df.copy()
    ema12 = d['close'].ewm(span=12, adjust=False).mean()
    ema26 = d['close'].ewm(span=26, adjust=False).mean()
    d['MACD_lag'] = (ema12 - ema26).shift(1)
    
    delta = d['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI_lag'] = (100 - (100 / (1 + gain / (loss + 1e-9)))).shift(1)
    
    d['Vol_lag'] = d['close'].rolling(20).std().shift(1)
    d['Ret_lag'] = d['close'].pct_change().shift(1)
    
    return d.dropna(), ['MACD_lag', 'RSI_lag', 'Vol_lag', 'Ret_lag']

def evaluate(symbol, tf_val, hold, sl, thr):
    rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, BARS_TOTAL)
    if rates is None or len(rates) < 2500: return -1
    
    df_raw = pd.DataFrame(rates)
    df, feats = get_features(df_raw)
    
    test_len = 2000
    train_df = df.iloc[:-test_len].copy()
    test_df = df.iloc[-test_len:].copy()
    
    # Учитель
    y_vals = gaussian_filter1d(train_df['close'].values, SIGMA, order=1)
    train_df['Target'] = np.where(y_vals > 0, 1, 0)
    train_final = train_df.iloc[:-LOOKAHEAD].dropna()
    
    if train_final.empty: return -1
    
    model = lgb.LGBMClassifier(n_estimators=50, verbosity=-1, min_child_samples=10)
    model.fit(train_final[feats], train_final['Target'])
    probs = model.predict_proba(test_df[feats])[:, 1]
    
    # Симуляция (с учетом SL)
    pos = 0; h_cnt = 0; pnl = []; trades = 0
    test_rets = test_df['ret' if 'ret' in test_df else 'Ret_lag'].values # используем доходность
    close_prices = test_df['close'].values
    entry_price = 0
    
    point = mt5.symbol_info(symbol).point
    sl_val = sl * point

    for i in range(len(test_df)-1):
        p = probs[i]
        curr_price = close_prices[i]
        
        if pos == 0:
            if p > thr: 
                pos = 1; h_cnt = 0; trades += 1; entry_price = curr_price
            elif p < (1-thr): 
                pos = -1; h_cnt = 0; trades += 1; entry_price = curr_price
        else:
            h_cnt += 1
            # Проверка SL
            if pos == 1 and (curr_price <= entry_price - sl_val): pos = 0
            elif pos == -1 and (curr_price >= entry_price + sl_val): pos = 0
            # Тайм-аут или разворот
            elif h_cnt >= hold or (pos == 1 and p < (1-thr)) or (pos == -1 and p > thr):
                pos = 0
        
        pnl.append(pos * test_rets[i+1])
    
    res = np.sum(pnl) - (trades * SPREAD_COST)
    sharpe = (np.mean(pnl) / (np.std(pnl) + 1e-9)) * np.sqrt(252)
    return sharpe if res > 0 else -1

if not mt5.initialize(): exit()

existing_symbols = [s.name for s in mt5.symbols_get()]
valid_symbols = [s for s in SYMBOLS if s in existing_symbols]

best_configs = []

for sym in valid_symbols:
    for tf_name, tf_val in TIMEFRAMES.items():
        print(f"Поиск для {sym} {tf_name}...")
        
        def objective(trial):
            hold = trial.suggest_int('hold', 5, 40) # Увеличил время удержания
            sl = trial.suggest_int('sl', 1000, 6000) # Подбор стопа (в пунктах)
            thr = trial.suggest_float('thr', 0.55, 0.68) # Более строгий порог
            return evaluate(sym, tf_val, hold, sl, thr)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS_PER_COMBO)
        
        if study.best_value > 0.35:
            best_configs.append({
                'symbol': sym, 'tf': tf_name, 
                'hold': study.best_params['hold'],
                'sl': study.best_params['sl'],
                'thr': study.best_params['thr'],
                'sharpe': study.best_value
            })

if best_configs:
    pd.DataFrame(best_configs).to_csv("best_configs_BTC.csv", index=False)
    print("Генерация завершена. Файл best_configs_BTC.csv готов.")
mt5.shutdown()