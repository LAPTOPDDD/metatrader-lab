# train_v5_optuna.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import os
import optuna
from config_v5 import *

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not mt5.initialize():
    print("Ошибка инициализации MT5")
    sys.exit()

def add_advanced_features(df):
    """Генерация каузальных признаков (только прошлое)"""
    # MACD
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_lag_1'] = df['MACD'].shift(1)
    df['MACD_lag_2'] = df['MACD'].shift(2)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_lag_1'] = df['RSI'].shift(1)

    # Returns
    df['Ret_1'] = df['close'].pct_change(1)
    df['Ret_3'] = df['close'].pct_change(3)
    df['Ret_5'] = df['close'].pct_change(5)

    # Bollinger Bands
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['BB_pos'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df['BB_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-9)

    # ATR Relative
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_rel'] = tr.rolling(14).mean() / df['close']

    features = ['MACD_lag_1', 'MACD_lag_2', 'RSI_lag_1', 'Ret_1', 'Ret_3', 'Ret_5', 'BB_pos', 'BB_width', 'ATR_rel']
    return df, features


def run_walk_forward_for_strategy(symbol, timeframe, hold_bars, sl_mult):
    """Честный Walk-Forward: Учитель пересчитывается каузально для каждого шага"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, BARS_TOTAL)
    if rates is None or len(rates) < BARS_TOTAL // 2:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[~df.index.dayofweek.isin([5, 6])].copy()

    # Считаем фичи сразу (они каузальны, будущее не видят)
    df, features = add_advanced_features(df)
    df_clean = df.dropna().copy()

    if len(df_clean) < (TOTAL_OOS_BARS + 500):
        return None

    oos_pnl_chain = []

    # === WALK-FORWARD LOOP ===
    for s in range(N_STEPS):
        # Определяем границы текущего окна
        test_start_idx = len(df_clean) - (N_STEPS - s) * STEP_BARS
        test_end_idx = test_start_idx + STEP_BARS
        
        # 1. Выделяем историю ДО начала теста
        train_df_raw = df_clean.iloc[:test_start_idx].copy()
        
        # 2. РАСЧЕТ УЧИТЕЛЯ (Гаусс видит только текущую историю)
        # Sigma=2, как в боте. 
        train_df_raw['Conv'] = gaussian_filter1d(train_df_raw['close'].values, sigma=2, mode='nearest')
        train_df_raw['Target_Binary'] = (train_df_raw['Conv'].diff() > 0).astype(int)
        
        # 3. ПРИНЦИП ГЛУБОКОЙ ИСТОРИИ
        # Ученик (LGBM) учится только на данных с отступом 5 баров от края,
        # где Гаусс уже стабилен и не перерисуется.
        train_df_final = train_df_raw.iloc[:-5].dropna()
        
        if len(train_df_final) < 200: continue

        # 4. ОБУЧЕНИЕ УЧЕНИКА
        model = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
        )
        model.fit(train_df_final[features], train_df_final['Target_Binary'])

        # 5. ТЕСТ (Только аппроксиматор на "новом" куске данных)
        # Берем тест + 1 бар сзади для корректного расчета первого входа
        test_df = df_clean.iloc[test_start_idx : test_end_idx].copy()
        
        probs = model.predict_proba(test_df[features])[:, 1]
        signals = np.zeros(len(probs))
        signals[probs >= PROBA_BUY] = 1
        signals[probs <= PROBA_SELL] = -1

        # Симуляция торговли (аналогично логике в боте)
        positions = np.zeros(len(signals))
        pos = 0
        hold_counter = 0
        entry_price = 0.0
        atr_points = 0.0

        for i in range(len(signals)):
            price_close = test_df['close'].iloc[i]
            price_low = test_df['low'].iloc[i]
            price_high = test_df['high'].iloc[i]
            
            # Текущий ATR для стопа
            curr_atr = test_df['ATR_rel'].iloc[i] * price_close

            # Проверка SL (более реалистично - по Low/High)
            if pos == 1:
                if price_low <= (entry_price - sl_mult * atr_points):
                    pos = 0
            elif pos == -1:
                if price_high >= (entry_price + sl_mult * atr_points):
                    pos = 0

            # Логика входа/выхода
            sig = signals[i]
            if pos == 0:
                if sig != 0:
                    pos = sig
                    hold_counter = 1
                    entry_price = price_close
                    atr_points = curr_atr
            else:
                hold_counter += 1
                if hold_counter >= hold_bars or (sig != 0 and sig != pos):
                    # Реверс или выход по времени
                    if sig != 0 and sig != pos:
                        pos = sig
                        entry_price = price_close
                        atr_points = curr_atr
                        hold_counter = 1
                    else:
                        pos = 0
                        hold_counter = 0

            positions[i] = pos

        # Считаем PnL шага
        test_df['Position'] = positions
        test_df['Returns'] = test_df['close'].pct_change().fillna(0)
        # Сдвиг, так как позиция открывается в конце бара i и работает на баре i+1
        step_pnl = (test_df['Position'].shift(1) * test_df['Returns']).fillna(0).values
        oos_pnl_chain.extend(step_pnl)

    if not oos_pnl_chain: return None
    
    full_oos_pnl = np.array(oos_pnl_chain)
    full_oos_equity = np.cumsum(full_oos_pnl)
    pnl_std = np.std(full_oos_pnl)
    sharpe = (np.mean(full_oos_pnl) / (pnl_std + 1e-9)) * np.sqrt(252) if pnl_std > 0 else 0

    return {
        'symbol': symbol,
        'timeframe': TIMEFRAME_NAMES[timeframe],
        'hold_bars': hold_bars,
        'sl_mult': sl_mult,
        'wf_return': full_oos_equity[-1],
        'wf_sharpe_sl': sharpe,
        'wf_equity': full_oos_equity,
        'wf_pnl': full_oos_pnl,
        'std': pnl_std if pnl_std > 0 else 1e-6
    }

# =============================================================
# ОСТАЛЬНАЯ ЧАСТЬ (Optuna и Ансамбль) ОСТАЕТСЯ БЕЗ ИЗМЕНЕНИЙ
# =============================================================
all_results = []

def objective(trial):
    symbol = trial.suggest_categorical('symbol', SYMBOLS)
    tf_name = trial.suggest_categorical('timeframe', ['M30', 'H1', 'H4'])
    tf = {'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4}[tf_name]
    
    hold_bars = trial.suggest_int('hold_bars', 1, 15)
    sl_mult = trial.suggest_float('sl_mult', SL_MULT_MIN, SL_MULT_MAX, step=SL_MULT_STEP)
    
    res = run_walk_forward_for_strategy(symbol, tf, hold_bars, sl_mult)
    if res is None:
        return -1.0
    
    all_results.append(res)
    return res['wf_sharpe_sl']

print("=== V5 Честный Walk-Forward (Учитель-Ученик) ===")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100) # Для начала 100 итераций

# ... (код сохранения в CSV и построения графиков из предыдущей версии) ...