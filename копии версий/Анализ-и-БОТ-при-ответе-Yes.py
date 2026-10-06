import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==================== НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ ====================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
MAGIC = 100006            # Уникальный ID бота
LOT = 0.1                 # Объем сделки
SIGMA = 1.0               # Параметр Учителя (Гаусс)
LOOKAHEAD = int(4 * SIGMA)# Математический отступ (4 сигмы)
RETRAIN_EVERY = 5         # Переобучать модель каждые N баров
BARS_FOR_BACKTEST = 400   # Глубина анализа на графике
HISTORY_TOTAL = 10000     # Всего баров для обучения

# Параметры стратегии
THR = 0.38                # Порог вероятности (уверенность модели)
HOLD_BARS = 4             # Макс. время удержания сделки
STOP_LOSS_PIPS = 50       # Стоп-лосс в пунктах
# ================================================================

def get_features(df):
    """Расчет казуальных признаков (только прошлое)"""
    d = df.copy()
    # MACD
    d['MACD'] = d['close'].ewm(span=12).mean() - d['close'].ewm(span=26).mean()
    for i in range(1, 4): 
        d[f'MACD_lag_{i}'] = d['MACD'].shift(i)
    # Momentum
    for k in [6, 12]: 
        d[f'mom_{k}'] = d['close'].pct_change(k)
    # RSI
    delta = d['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
    d['rsi_14'] = 100 - 100 / (1 + gain / (loss + 1e-9))
    # Shadows & Shadows
    rng = (d['high'] - d['low']).replace(0, np.nan)
    d['body'] = (d['close'] - d['open']) / rng
    # Volatility
    d['hl_range'] = (d['high'] - d['low']) / d['close']
    d['vol_norm'] = d['tick_volume'] / d['tick_volume'].rolling(20).mean()
    return d.dropna()

def build_model(data_subset):
    """Обучение Ученика (LGBM) на Учителе (Гаусс) с отступом"""
    d = data_subset.copy()
    # Целевая переменная: наклон Гаусса
    d['Conv_Deriv'] = gaussian_filter1d(d['close'].values, SIGMA, order=1)
    d['Target'] = np.where(d['Conv_Deriv'] > 0, 1, 0)
    
    # Отступаем 4 сигмы от края, чтобы таргет был стабильным
    train_data = d.iloc[:-LOOKAHEAD].dropna()
    
    # Список фичей (исключаем служебные колонки)
    feats = [c for c in train_data.columns if c not in [
        'open','high','low','close','tick_volume','spread','real_volume','Target','Conv_Deriv','time'
    ]]
    
    m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, verbosity=-1, random_state=42)
    m.fit(train_data[feats], train_data['Target'])
    return m, feats

def get_filling_mode(symbol):
    """Автоматическое определение режима заполнения ордеров (Filling Mode)"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return mt5.ORDER_FILLING_IOC
    
    flags = symbol_info.filling_mode
    if flags & mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    elif flags & mt5.SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

def open_trade(order_type, prob):
    """Открытие сделки с динамическим определением Filling Mode"""
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None: return
    
    point = symbol_info.point
    price = mt5.symbol_info_tick(SYMBOL).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).bid
    sl = price - STOP_LOSS_PIPS * point if order_type == mt5.ORDER_TYPE_BUY else price + STOP_LOSS_PIPS * point
    
    fill_mode = get_filling_mode(SYMBOL)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": order_type,
        "price": price,
        "sl": sl,
        "magic": MAGIC,
        "comment": f"P={prob:.2f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": fill_mode,
    }
    res = mt5.order_send(request)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Ошибка открытия: {res.comment} (Код: {res.retcode})")
    else:
        print(f"Сделка {'BUY' if order_type==0 else 'SELL'} открыта! Вероятность: {prob:.2f}")

def close_position(p):
    """Закрытие позиции"""
    tick = mt5.symbol_info_tick(SYMBOL)
    fill_mode = get_filling_mode(SYMBOL)
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": p.volume,
        "type": mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": p.ticket,
        "price": tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask,
        "magic": MAGIC,
        "comment": "Close Position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": fill_mode,
    }
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Ошибка закрытия: {res.comment}")

def run_hybrid_bot():
    if not mt5.initialize():
        print("Ошибка инициализации MT5")
        return

    print(f"--- АНАЛИЗ {SYMBOL} ---")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, HISTORY_TOTAL)
    if rates is None:
        print("Не удалось получить данные")
        return

    df_full = pd.DataFrame(rates)
    df_feats = get_features(df_full)
    
    # 1. ЧЕСТНЫЙ БЭКТЕСТ ДЛЯ ПРОВЕРКИ
    test_start_idx = len(df_feats) - BARS_FOR_BACKTEST
    test_df = df_feats.iloc[test_start_idx:].copy()
    
    print("Запуск симуляции последних баров...")
    model, feats = build_model(df_feats.iloc[:test_start_idx])
    
    probs = []
    for i in range(len(test_df)):
        if i % RETRAIN_EVERY == 0 and i > 0:
            # Инкрементальное обучение
            model, feats = build_model(df_feats.iloc[:test_start_idx + i])
        
        row = test_df.iloc[i:i+1][feats]
        probs.append(model.predict_proba(row)[0][1])
    
    test_df['P_long'] = probs
    # Вход на следующем баре после сигнала (shift 1)
    test_df['Sig'] = np.where(test_df['P_long'] > THR, 1, np.where(test_df['P_long'] < (1-THR), -1, 0))
    test_df['Next_Ret'] = test_df['close'].pct_change().shift(-1)
    test_df['Equity'] = (test_df['Sig'] * test_df['Next_Ret']).fillna(0).cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['Equity'].values, color='green', lw=2)
    plt.title(f"Честный бэктест {SYMBOL} (Последние {BARS_FOR_BACKTEST} баров)")
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.show(block=False)
    
    ans = input("Эквити нравится? Запустить LIVE-торговлю? (y/n): ")
    if ans.lower() != 'y':
        print("Остановка.")
        mt5.shutdown()
        return

    # 2. ЦИКЛ ЖИВОЙ ТОРГОВЛИ
    print(f"\nБОТ ЗАПУЩЕН. Маджик: {MAGIC}, Лот: {LOT}")
    last_processed_time = 0
    
    try:
        while True:
            # Получаем 2 последних бара, чтобы поймать момент открытия нового
            r = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 2)
            if r is None: continue
            
            current_bar_time = r[-1]['time']
            
            if current_bar_time != last_processed_time:
                # Наступил новый бар!
                last_processed_time = current_bar_time
                
                # Обновляем данные
                all_r = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, HISTORY_TOTAL)
                df_live = get_features(pd.DataFrame(all_r))
                
                # Переобучаем модель на свежих данных
                model, feats = build_model(df_live)
                
                # Прогноз по последнему закрытому бару (индекс -1)
                last_row = df_live.iloc[-1:][feats]
                prob = model.predict_proba(last_row)[0][1]
                
                # Информация о текущих позициях
                positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC)
                
                # Логика закрытия
                if positions:
                    p = positions[0]
                    # Время жизни в барах
                    bars_held = (current_bar_time - p.time) / (TIMEFRAME * 60)
                    
                    sig_long = prob > THR
                    sig_short = prob < (1 - THR)
                    
                    should_close = False
                    if bars_held >= HOLD_BARS: should_close = True # Время вышло
                    if p.type == mt5.ORDER_TYPE_BUY and sig_short: should_close = True # Переворот
                    if p.type == mt5.ORDER_TYPE_SELL and sig_long: should_close = True # Переворот
                    
                    if should_close:
                        close_position(p)
                        # Ждем небольшую паузу, чтобы терминал обновил список позиций
                        time.sleep(0.5)
                        positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC)

                # Логика открытия
                if not positions:
                    if prob > THR:
                        open_trade(mt5.ORDER_TYPE_BUY, prob)
                    elif prob < (1 - THR):
                        open_trade(mt5.ORDER_TYPE_SELL, prob)
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {SYMBOL} M15 | Прогноз: {prob:.2f} | Сделок: {len(positions)}")

            time.sleep(1) # Проверка каждую секунду
            
    except KeyboardInterrupt:
        print("\nОстановка пользователем.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    run_hybrid_bot()