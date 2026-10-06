import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import time
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ====================
CONFIG_FILE = "best_configs.csv"
LOT_PER_AGENT = 0.01      # Минимальный лот для каждой стратегии
SIGMA = 2.0               # Параметр Учителя
LOOKAHEAD = 8             # Математический отступ (4 * SIGMA)
HISTORY_BARS = 5000       # Глубина истории для обучения агентов
RETRAIN_BARS = 50         # Частота переобучения (в барах)

# ==================== ФУНКЦИИ МАТЕМАТИКИ И ТОРГОВЛИ ====================

def get_features(df):
    """Единая функция расчета признаков (должна совпадать со сканером)"""
    d = df.copy()
    # MACD лаг 1
    ema12 = d['close'].ewm(span=12, adjust=False).mean()
    ema26 = d['close'].ewm(span=26, adjust=False).mean()
    d['MACD_lag'] = (ema12 - ema26).shift(1)
    
    # RSI лаг 1
    delta = d['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI_lag'] = (100 - (100 / (1 + gain / (loss + 1e-9)))).shift(1)
    
    # Volatility лаг 1
    d['Vol_lag'] = d['close'].rolling(20).std().shift(1)
    
    # Изменение цены лаг 1
    d['Ret_lag'] = d['close'].pct_change().shift(1)
    
    return d.dropna(), ['MACD_lag', 'RSI_lag', 'Vol_lag', 'Ret_lag']

def get_filling_mode(symbol):
    """Универсальное определение режима заполнения ордеров"""
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    
    # Получаем значение режима заполнения из свойств символа
    fill_mode = info.filling_mode
    
    # Стандартные константы MT5: 
    # SYMBOL_FILLING_FOK = 1
    # SYMBOL_FILLING_IOC = 2
    
    # Пытаемся найти константы в модуле, если их нет - используем числа
    fok_flag = getattr(mt5, 'SYMBOL_FILLING_FOK', 1)
    ioc_flag = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)
    
    if fill_mode & fok_flag:
        return mt5.ORDER_FILLING_FOK
    elif fill_mode & ioc_flag:
        return mt5.ORDER_FILLING_IOC
    else:
        # Если ваш брокер не поддерживает FOK/IOC (например, на ECN), 
        # возвращаем RETURN (для Market Execution)
        return mt5.ORDER_FILLING_RETURN

# ==================== КЛАСС ТОРГОВОГО АГЕНТА ====================

class TradingAgent:
    def __init__(self, id, row):
        self.id = id
        self.symbol = row['symbol']
        self.tf_name = row['tf']
        self.tf_val = getattr(mt5, f"TIMEFRAME_{row['tf']}")
        self.hold_bars = int(row['hold'])
        self.sl_pips = int(row['sl'])
        self.thr = float(row['thr'])
        self.magic = 200000 + id
        
        self.model = None
        self.features = []
        self.last_bar_time = 0
        self.bars_since_train = 0
        self.bars_held = 0
        
        print(f"Инициализирован агент {self.id}: {self.symbol} {self.tf_name}")
        self.retrain()

    def retrain(self):
        """Обучение Ученика на Учителе с учетом математического отступа"""
        rates = mt5.copy_rates_from_pos(self.symbol, self.tf_val, 0, HISTORY_BARS)
        if rates is None: return
        
        df = pd.DataFrame(rates)
        # Учитель (Гаусс)
        df['Conv_Deriv'] = gaussian_filter1d(df['close'].values, SIGMA, order=1)
        df['Target'] = np.where(df['Conv_Deriv'] > 0, 1, 0)
        
        df_proc, self.features = get_features(df)
        
        # Математический отступ (LOOKAHEAD)
        train_data = df_proc.iloc[:-LOOKAHEAD].dropna()
        
        self.model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
        self.model.fit(train_data[self.features], train_data['Target'])
        self.bars_since_train = 0
        print(f"[{self.symbol}] Модель переобучена успешно.")

    def trade(self):
        # Проверяем новый бар
        rates = mt5.copy_rates_from_pos(self.symbol, self.tf_val, 0, 2)
        if rates is None or rates[-1]['time'] == self.last_bar_time:
            return
        
        self.last_bar_time = rates[-1]['time']
        self.bars_since_train += 1
        
        # Нужно ли переобучение?
        if self.bars_since_train >= RETRAIN_BARS:
            self.retrain()

        # 1. Получаем прогноз
        all_r = mt5.copy_rates_from_pos(self.symbol, self.tf_val, 0, 100)
        df_live, _ = get_features(pd.DataFrame(all_r))
        last_row = df_live.iloc[-1:][self.features]
        prob = self.model.predict_proba(last_row)[0][1]
        
        # 2. Проверка текущих позиций
        pos = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        
        if pos:
            p = pos[0]
            self.bars_held += 1
            
            # Условия закрытия
            should_close = False
            if self.bars_held >= self.hold_bars: should_close = True # Тайм-аут
            if p.type == 0 and prob < (1 - self.thr): should_close = True # Разворот
            if p.type == 1 and prob > self.thr: should_close = True # Разворот
            
            if should_close:
                self.close_order(p)
                pos = None # Позиция закрыта
        
        # 3. Условия открытия
        if not pos:
            if prob >= self.thr:
                self.open_order(mt5.ORDER_TYPE_BUY, prob)
            elif prob <= (1 - self.thr):
                self.open_order(mt5.ORDER_TYPE_SELL, prob)

    def open_order(self, order_type, prob):
        symbol_info = mt5.symbol_info(self.symbol)
        price = mt5.symbol_info_tick(self.symbol).ask if order_type == 0 else mt5.symbol_info_tick(self.symbol).bid
        point = symbol_info.point
        sl = price - self.sl_pips * point if order_type == 0 else price + self.sl_pips * point
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": LOT_PER_AGENT,
            "type": order_type,
            "price": price,
            "sl": sl,
            "magic": self.magic,
            "comment": f"V6 Prob:{prob:.2f}",
            "type_filling": get_filling_mode(self.symbol),
            "type_time": mt5.ORDER_TIME_GTC,
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            self.bars_held = 0
            print(f"[{self.symbol}] ОТКРЫТ {'BUY' if order_type==0 else 'SELL'} (P={prob:.2f})")
        else:
            print(f"[{self.symbol}] Ошибка открытия: {res.comment}")

    def close_order(self, p):
        tick = mt5.symbol_info_tick(self.symbol)
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": p.volume,
            "type": mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": p.ticket,
            "price": tick.bid if p.type == 0 else tick.ask,
            "magic": self.magic,
            "type_filling": get_filling_mode(self.symbol),
            "type_time": mt5.ORDER_TIME_GTC,
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[{self.symbol}] Позиция закрыта (Time-out/Signal)")

# ==================== ЗАПУСК ПОРТФЕЛЯ ====================

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE):
        print(f"Файл {CONFIG_FILE} не найден! Сначала запустите сканер.")
        exit()

    if not mt5.initialize():
        print("Ошибка инициализации MT5")
        exit()

    # Читаем конфиги
    configs_df = pd.read_csv(CONFIG_FILE)
    if configs_df.empty:
        print("Конфиги пусты. Сканеру не удалось найти прибыльные пары.")
        mt5.shutdown()
        exit()

    # Создаем агентов
    portfolio = []
    for index, row in configs_df.iterrows():
        try:
            agent = TradingAgent(index, row)
            portfolio.append(agent)
        except Exception as e:
            print(f"Не удалось запустить агента для {row['symbol']}: {e}")

    print(f"\nВсего запущено агентов: {len(portfolio)}")
    print("Бот V6 Multi-Agent работает... (Ctrl+C для остановки)")

    try:
        while True:
            for agent in portfolio:
                agent.trade()
            time.sleep(1) # Проверка раз в секунду
    except KeyboardInterrupt:
        print("\nБот остановлен пользователем.")
    finally:
        mt5.shutdown()