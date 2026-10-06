import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time
from datetime import datetime
import warnings
import sys

warnings.filterwarnings('ignore')

# ====================================================================
#                     НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ
# ====================================================================

# Торговые параметры
SYMBOL = "EURUSD"                 # Торговый инструмент
TIMEFRAME = mt5.TIMEFRAME_H1      # Таймфрейм
MAGIC = 100006                    # Уникальный ID бота
LOT = 0.1                         # Объем сделки

# Параметры модели (Учитель-Ученик)
SIGMA = 1.0                       # Параметр гауссовского фильтра
LOOKAHEAD = int(4 * SIGMA)        # Отступ от края (4 сигмы)
RETRAIN_EVERY = 5                 # Переобучать каждые N баров

# Параметры стратегии
THR = 0.38                        # Порог вероятности (уверенность модели)
HOLD_BARS = 4                     # Макс. время удержания (в барах)
STOP_LOSS_PIPS = 50               # Стоп-лосс в пунктах (будет скорректирован)

# Параметры бэктеста
BARS_FOR_BACKTEST = 400           # Глубина тестирования
HISTORY_TOTAL = 10000             # Всего баров для обучения

# ====================================================================
#                     ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ====================================================================

def get_timeframe_seconds(tf):
    """Получить длительность таймфрейма в секундах"""
    timeframe_map = {
        mt5.TIMEFRAME_M1: 60,
        mt5.TIMEFRAME_M2: 120,
        mt5.TIMEFRAME_M3: 180,
        mt5.TIMEFRAME_M4: 240,
        mt5.TIMEFRAME_M5: 300,
        mt5.TIMEFRAME_M6: 360,
        mt5.TIMEFRAME_M10: 600,
        mt5.TIMEFRAME_M12: 720,
        mt5.TIMEFRAME_M15: 900,
        mt5.TIMEFRAME_M20: 1200,
        mt5.TIMEFRAME_M30: 1800,
        mt5.TIMEFRAME_H1: 3600,
        mt5.TIMEFRAME_H2: 7200,
        mt5.TIMEFRAME_H3: 10800,
        mt5.TIMEFRAME_H4: 14400,
        mt5.TIMEFRAME_H6: 21600,
        mt5.TIMEFRAME_H8: 28800,
        mt5.TIMEFRAME_H12: 43200,
        mt5.TIMEFRAME_D1: 86400,
        mt5.TIMEFRAME_W1: 604800,
        mt5.TIMEFRAME_MN1: 2592000,
    }
    return timeframe_map.get(tf, 3600)  # По умолчанию H1

def get_features(df):
    """Расчет признаков (только прошлые данные)"""
    d = df.copy()
    
    # MACD
    d['MACD'] = d['close'].ewm(span=12, adjust=False).mean() - d['close'].ewm(span=26, adjust=False).mean()
    for i in range(1, 4):
        d[f'MACD_lag_{i}'] = d['MACD'].shift(i)
    
    # Momentum
    for k in [6, 12, 24]:
        d[f'mom_{k}'] = d['close'].pct_change(k)
    
    # RSI
    delta = d['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d['rsi_14'] = 100 - 100 / (1 + gain / (loss + 1e-9))
    
    # Body/Shadows
    rng = (d['high'] - d['low']).replace(0, np.nan)
    d['body'] = (d['close'] - d['open']) / rng
    d['upper_shadow'] = (d['high'] - d[['open', 'close']].max(axis=1)) / rng
    d['lower_shadow'] = (d[['open', 'close']].min(axis=1) - d['low']) / rng
    
    # Volatility
    d['hl_range'] = (d['high'] - d['low']) / d['close']
    d['vol_norm'] = d['tick_volume'] / d['tick_volume'].rolling(20).mean()
    d['atr_14'] = d['hl_range'].rolling(14).mean()
    
    # Price position
    d['close_ma_20'] = d['close'] / d['close'].rolling(20).mean()
    d['close_ma_50'] = d['close'] / d['close'].rolling(50).mean()
    
    return d.dropna()

def build_model(data_subset, exclude_last=True):
    """
    Обучение модели LightGBM
    
    Parameters:
    data_subset: DataFrame с данными
    exclude_last: bool - отступать ли LOOKAHEAD баров от правого края
    """
    d = data_subset.copy()
    
    # Целевая переменная: производная гауссовского фильтра
    d['Conv_Deriv'] = gaussian_filter1d(d['close'].values, SIGMA, order=1)
    d['Target'] = np.where(d['Conv_Deriv'] > 0, 1, 0)
    
    # Отступ от правого края для избежания артефактов свертки
    if exclude_last:
        train_data = d.iloc[:-LOOKAHEAD].dropna()
    else:
        train_data = d.dropna()
    
    if len(train_data) < 100:
        print(f"⚠️ Недостаточно данных для обучения: {len(train_data)} баров")
        return None, None
    
    # Список признаков (исключаем служебные колонки)
    exclude_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 
                   'real_volume', 'Target', 'Conv_Deriv', 'time']
    feats = [c for c in train_data.columns if c not in exclude_cols]
    
    # Обучение модели
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(train_data[feats], train_data['Target'])
    return model, feats

def get_filling_mode(symbol):
    """Определение режима заполнения ордеров"""
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

def get_valid_stop_level(symbol, price, sl_price, is_buy):
    """
    Корректировка стоп-лосса с учетом минимальных расстояний, разрешенных брокером
    
    Returns:
    float: скорректированная цена стоп-лосса
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return sl_price
    
    point = symbol_info.point
    min_stop = symbol_info.trade_tick_size * 10  # Минимальное расстояние в пунктах
    
    if is_buy:
        # Для BUY стоп должен быть ниже цены входа
        min_sl_price = price - min_stop
        if sl_price > min_sl_price:
            sl_price = min_sl_price
    else:
        # Для SELL стоп должен быть выше цены входа
        min_sl_price = price + min_stop
        if sl_price < min_sl_price:
            sl_price = min_sl_price
    
    # Округляем до размера тика
    tick_size = symbol_info.trade_tick_size
    sl_price = round(sl_price / tick_size) * tick_size
    
    return sl_price

def open_trade(order_type, prob):
    """Открытие сделки с учетом правил брокера"""
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"❌ Не удалось получить информацию по {SYMBOL}")
        return False
    
    # Проверка возможности торговли
    if not symbol_info.visible:
        print(f"❌ Символ {SYMBOL} не виден в Market Watch")
        return False
    
    if not symbol_info.trade_mode in [mt5.SYMBOL_TRADE_MODE_FULL, mt5.SYMBOL_TRADE_MODE_LONGONLY, mt5.SYMBOL_TRADE_MODE_SHORTONLY]:
        print(f"❌ Торговля по {SYMBOL} запрещена")
        return False
    
    point = symbol_info.point
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("❌ Не удалось получить текущую цену")
        return False
    
    # Цена входа
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    
    # Расчет стоп-лосса
    sl_distance = STOP_LOSS_PIPS * point
    raw_sl = price - sl_distance if order_type == mt5.ORDER_TYPE_BUY else price + sl_distance
    
    # Корректировка стопа согласно правилам брокера
    sl = get_valid_stop_level(SYMBOL, price, raw_sl, order_type == mt5.ORDER_TYPE_BUY)
    
    # Проверка минимального и максимального объема
    if LOT < symbol_info.volume_min or LOT > symbol_info.volume_max:
        print(f"⚠️ Объем {LOT} вне допустимого диапазона [{symbol_info.volume_min}, {symbol_info.volume_max}]")
        return False
    
    # Шаг изменения объема
    if LOT % symbol_info.volume_step != 0:
        LOT_adjusted = round(LOT / symbol_info.volume_step) * symbol_info.volume_step
        print(f"⚠️ Объем скорректирован с {LOT} до {LOT_adjusted}")
        LOT_use = LOT_adjusted
    else:
        LOT_use = LOT
    
    fill_mode = get_filling_mode(SYMBOL)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_use,
        "type": order_type,
        "price": price,
        "sl": sl,
        "magic": MAGIC,
        "comment": f"P={prob:.2f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": fill_mode,
    }
    
    # Отправка ордера
    res = mt5.order_send(request)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Ошибка открытия: {res.comment} (Код: {res.retcode})")
        return False
    else:
        print(f"✅ {'BUY' if order_type==0 else 'SELL'} открыта! Объем: {LOT_use}, Цена: {price:.5f}, SL: {sl:.5f}, Вероятность: {prob:.2f}")
        return True

def close_position(position):
    """Закрытие позиции"""
    if position is None:
        return False
    
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("❌ Не удалось получить текущую цену")
        return False
    
    fill_mode = get_filling_mode(SYMBOL)
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask,
        "magic": MAGIC,
        "comment": "Close Position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": fill_mode,
    }
    
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Ошибка закрытия: {res.comment} (Код: {res.retcode})")
        return False
    else:
        profit = position.profit if hasattr(position, 'profit') else 0
        print(f"✅ Позиция закрыта. Profit: {profit:.2f}")
        return True

def get_bars_count(time_diff_seconds, tf_seconds):
    """Количество баров между двумя временными метками"""
    return int(time_diff_seconds / tf_seconds)

# ====================================================================
#                     БЭКТЕСТ
# ====================================================================

def run_backtest(df_feats):
    """Запуск честного бэктеста"""
    print("\n" + "="*60)
    print("ЗАПУСК БЭКТЕСТА")
    print("="*60)
    
    # Определяем тестовое окно с учетом отступа
    total_bars = len(df_feats)
    test_start_idx = total_bars - BARS_FOR_BACKTEST - LOOKAHEAD
    
    if test_start_idx < LOOKAHEAD:
        print("❌ Недостаточно данных для бэктеста")
        return None
    
    # Данные для теста (начинаем с отступом LOOKAHEAD)
    test_start_real = test_start_idx + LOOKAHEAD
    test_df = df_feats.iloc[test_start_real:].copy()
    test_df = test_df.reset_index(drop=True)
    
    print(f"📊 Тестовых баров: {len(test_df)}")
    print(f"📊 Отступ от края: {LOOKAHEAD} баров")
    
    # Начальное обучение
    initial_train = df_feats.iloc[:test_start_idx + LOOKAHEAD].copy()
    model, feats = build_model(initial_train, exclude_last=True)
    
    if model is None:
        print("❌ Не удалось обучить начальную модель")
        return None
    
    print(f"✅ Модель обучена. Признаков: {len(feats)}")
    
    # Бэктест
    probs = []
    signals = []
    trades = []
    equity = [0.0]
    position = None
    entry_price = 0
    entry_time = 0
    tf_seconds = get_timeframe_seconds(TIMEFRAME)
    
    for i in range(len(test_df)):
        # Переобучение каждые RETRAIN_EVERY баров
        if i % RETRAIN_EVERY == 0 and i > 0:
            available_data = df_feats.iloc[:test_start_idx + i + LOOKAHEAD].copy()
            model, feats = build_model(available_data, exclude_last=True)
            if model is None:
                print(f"⚠️ Ошибка переобучения на баре {i}")
                # Продолжаем со старой моделью
        
        # Прогноз для текущего бара
        row = test_df.iloc[i:i+1][feats]
        prob = model.predict_proba(row)[0][1]
        probs.append(prob)
        
        # Сигнал
        signal = 0
        if prob > THR:
            signal = 1  # BUY
        elif prob < (1 - THR):
            signal = -1  # SELL
        signals.append(signal)
        
        # Симуляция торговли
        current_price = test_df.iloc[i]['close']
        current_time = test_df.iloc[i]['time']
        
        # Проверка стоп-лосса (если есть позиция)
        if position is not None:
            # Проверяем, не сработал ли стоп-лосс
            point = 0.0001  # Для EURUSD
            sl_hit = False
            
            if position == 1:  # BUY
                if current_price <= entry_price - STOP_LOSS_PIPS * point:
                    sl_hit = True
                    exit_price = entry_price - STOP_LOSS_PIPS * point
            else:  # SELL
                if current_price >= entry_price + STOP_LOSS_PIPS * point:
                    sl_hit = True
                    exit_price = entry_price + STOP_LOSS_PIPS * point
            
            if sl_hit:
                # Закрываем по стопу
                profit = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
                profit = profit / entry_price * 100  # В процентах
                equity.append(equity[-1] + profit)
                trades.append({'type': 'BUY' if position == 1 else 'SELL', 
                              'entry': entry_price, 'exit': exit_price, 
                              'profit': profit, 'reason': 'SL'})
                position = None
                continue
            
            # Проверка времени удержания
            bars_held = get_bars_count(current_time - entry_time, tf_seconds)
            if bars_held >= HOLD_BARS:
                # Закрываем по времени
                profit = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                profit = profit / entry_price * 100
                equity.append(equity[-1] + profit)
                trades.append({'type': 'BUY' if position == 1 else 'SELL', 
                              'entry': entry_price, 'exit': current_price, 
                              'profit': profit, 'reason': 'Time'})
                position = None
                continue
            
            # Проверка сигнала на закрытие (переворот)
            if (position == 1 and signal == -1) or (position == -1 and signal == 1):
                profit = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                profit = profit / entry_price * 100
                equity.append(equity[-1] + profit)
                trades.append({'type': 'BUY' if position == 1 else 'SELL', 
                              'entry': entry_price, 'exit': current_price, 
                              'profit': profit, 'reason': 'Reverse'})
                position = None
        
        # Открытие новой позиции
        if position is None and signal != 0:
            position = signal
            entry_price = current_price
            entry_time = current_time
            
            # Имитация проскальзывания и спреда
            spread = 0.0001  # Типичный спред для EURUSD
            if position == 1:
                entry_price += spread / 2
            else:
                entry_price -= spread / 2
    
    # Закрываем последнюю позицию
    if position is not None:
        current_price = test_df.iloc[-1]['close']
        profit = (current_price - entry_price) if position == 1 else (entry_price - current_price)
        profit = profit / entry_price * 100
        equity.append(equity[-1] + profit)
        trades.append({'type': 'BUY' if position == 1 else 'SELL', 
                      'entry': entry_price, 'exit': current_price, 
                      'profit': profit, 'reason': 'End'})
    
    # Результаты
    results = {
        'equity': np.array(equity),
        'trades': trades,
        'probs': probs,
        'signals': signals,
        'total_trades': len(trades),
        'win_trades': sum(1 for t in trades if t['profit'] > 0),
        'total_profit': sum(t['profit'] for t in trades),
        'max_drawdown': np.min(equity) - 0,  # Относительно начального капитала
        'avg_profit': np.mean([t['profit'] for t in trades]) if trades else 0,
        'win_rate': sum(1 for t in trades if t['profit'] > 0) / len(trades) if trades else 0,
        'profit_factor': sum(t['profit'] for t in trades if t['profit'] > 0) / abs(sum(t['profit'] for t in trades if t['profit'] < 0)) if any(t['profit'] < 0 for t in trades) else float('inf')
    }
    
    return results

# ====================================================================
#                     ОСНОВНОЙ ЦИКЛ
# ====================================================================

def run_hybrid_bot():
    """Главная функция бота"""
    
    # Инициализация MT5
    if not mt5.initialize():
        print("❌ Ошибка инициализации MT5")
        return
    
    print(f"\n{'='*60}")
    print(f"ГИБРИДНЫЙ БОТ (УЧИТЕЛЬ-УЧЕНИК)")
    print(f"{'='*60}")
    print(f"Инструмент: {SYMBOL}")
    print(f"Таймфрейм: {TIMEFRAME}")
    print(f"Сигма: {SIGMA}, Отступ: {LOOKAHEAD} баров")
    print(f"Порог: {THR}, Стоп-лосс: {STOP_LOSS_PIPS} пунктов")
    print(f"Макс. удержание: {HOLD_BARS} баров")
    print(f"{'='*60}\n")
    
    # Получение данных
    print("📥 Загрузка исторических данных...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, HISTORY_TOTAL + LOOKAHEAD * 2)
    if rates is None or len(rates) < 100:
        print(f"❌ Не удалось получить данные: {mt5.last_error()}")
        mt5.shutdown()
        return
    
    df_full = pd.DataFrame(rates)
    print(f"✅ Загружено {len(df_full)} баров")
    
    # Расчет признаков
    print("🧮 Расчет признаков...")
    df_feats = get_features(df_full)
    print(f"✅ Признаков: {len(df_feats.columns) - 10}")  # Приблизительно
    
    # Бэктест
    results = run_backtest(df_feats)
    if results is None:
        mt5.shutdown()
        return
    
    # Визуализация результатов бэктеста
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print("="*60)
    print(f"Всего сделок: {results['total_trades']}")
    print(f"Прибыльных: {results['win_trades']} ({results['win_rate']*100:.1f}%)")
    print(f"Общая прибыль: {results['total_profit']:.2f}%")
    print(f"Средняя прибыль: {results['avg_profit']:.2f}%")
    print(f"Макс. просадка: {results['max_drawdown']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # График эквити
    plt.figure(figsize=(14, 7))
    plt.plot(results['equity'], color='green', lw=2, label='Equity')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.fill_between(range(len(results['equity'])), 0, results['equity'], 
                     where=results['equity'] >= 0, color='green', alpha=0.2)
    plt.fill_between(range(len(results['equity'])), 0, results['equity'], 
                     where=results['equity'] < 0, color='red', alpha=0.2)
    plt.title(f"Бэктест {SYMBOL} (последние {BARS_FOR_BACKTEST} баров)", fontsize=14)
    plt.xlabel("Бары")
    plt.ylabel("Эквити, %")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block=False)
    
    # Запрос на запуск в реальном режиме
    print("\n" + "="*60)
    ans = input("✅ Запустить LIVE-торговлю? (y/n): ")
    if ans.lower() != 'y':
        print("🛑 Остановка.")
        mt5.shutdown()
        return
    
    # ================================================================
    #                     LIVE-ТОРГОВЛЯ
    # ================================================================
    
    print("\n" + "="*60)
    print("ЗАПУСК LIVE-ТОРГОВЛИ")
    print("="*60)
    print(f"БОТ ЗАПУЩЕН. Маджик: {MAGIC}, Лот: {LOT}")
    print("Для остановки нажмите Ctrl+C")
    print("="*60 + "\n")
    
    last_processed_time = 0
    model = None
    feats = None
    tf_seconds = get_timeframe_seconds(TIMEFRAME)
    
    try:
        while True:
            # Получаем последние бары
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, HISTORY_TOTAL + LOOKAHEAD)
            if rates is None or len(rates) < 100:
                print(f"⚠️ Ошибка получения данных: {mt5.last_error()}")
                time.sleep(10)
                continue
            
            current_bar_time = rates[-1]['time']
            
            # Проверка нового бара
            if current_bar_time != last_processed_time:
                last_processed_time = current_bar_time
                
                # Обновление данных
                df_live = get_features(pd.DataFrame(rates))
                
                # Обучение модели с отступом от правого края
                train_data = df_live.iloc[:-LOOKAHEAD].copy()
                model, feats = build_model(train_data, exclude_last=False)
                
                if model is None:
                    print("⚠️ Ошибка обучения модели, пропускаем бар")
                    time.sleep(1)
                    continue
                
                # Прогноз для последнего закрытого бара
                # Берем бар на позиции -LOOKAHEAD-1 (то есть за LOOKAHEAD до правого края)
                pred_row = df_live.iloc[-LOOKAHEAD-1:-LOOKAHEAD][feats]
                if len(pred_row) == 0:
                    print("⚠️ Недостаточно данных для прогноза")
                    time.sleep(1)
                    continue
                
                prob = model.predict_proba(pred_row)[0][1]
                
                # Получение текущих позиций
                positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC)
                if positions is None:
                    positions = []
                
                # --- Логика закрытия ---
                if positions:
                    for pos in positions:
                        should_close = False
                        close_reason = ""
                        
                        # Время удержания
                        bars_held = get_bars_count(current_bar_time - pos.time, tf_seconds)
                        if bars_held >= HOLD_BARS:
                            should_close = True
                            close_reason = "Time"
                        
                        # Сигнал на закрытие
                        if pos.type == mt5.ORDER_TYPE_BUY and prob < (1 - THR):
                            should_close = True
                            close_reason = "Signal"
                        elif pos.type == mt5.ORDER_TYPE_SELL and prob > THR:
                            should_close = True
                            close_reason = "Signal"
                        
                        if should_close:
                            print(f"🔴 Закрытие {pos.type} по причине: {close_reason}")
                            close_position(pos)
                            time.sleep(0.5)
                
                # --- Логика открытия ---
                positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC)
                if positions is None or len(positions) == 0:
                    if prob > THR:
                        print(f"🟢 Сигнал BUY (P={prob:.2f})")
                        open_trade(mt5.ORDER_TYPE_BUY, prob)
                    elif prob < (1 - THR):
                        print(f"🔴 Сигнал SELL (P={prob:.2f})")
                        open_trade(mt5.ORDER_TYPE_SELL, prob)
                
                # Информация о состоянии
                positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC)
                pos_count = len(positions) if positions else 0
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Прогноз: {prob:.3f} | Позиций: {pos_count}")
            
            time.sleep(1)  # Проверка каждую секунду
            
    except KeyboardInterrupt:
        print("\n🛑 Остановка пользователем.")
        
        # Закрываем все позиции
        positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC)
        if positions:
            print("📢 Закрытие всех позиций...")
            for pos in positions:
                close_position(pos)
                time.sleep(0.3)
    
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🔚 Завершение работы бота")
        mt5.shutdown()

# ====================================================================
#                     ЗАПУСК
# ====================================================================

if __name__ == "__main__":
    run_hybrid_bot()