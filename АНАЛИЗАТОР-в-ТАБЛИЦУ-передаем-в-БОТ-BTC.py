import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import time
import datetime

# ==================== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ====================
CONFIG_FILE = "best_configs_BTC.csv"
LOT_PER_AGENT = 0.01      
SIGMA = 2.0               
LOOKAHEAD = 8             
HISTORY_BARS = 5000       
RETRAIN_INTERVAL_HOURS = 4 # Переобучение каждые 4 часа

def get_features(df):
    if df is None or len(df) < 30: return pd.DataFrame(), []
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

def get_filling_mode(symbol):
    info = mt5.symbol_info(symbol)
    if info is None: return mt5.ORDER_FILLING_FOK
    # Правильная проверка режимов исполнения
    if info.filling_mode & 1: return mt5.ORDER_FILLING_FOK
    if info.filling_mode & 2: return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

class TradingAgent:
    def __init__(self, id, row):
        self.id = id
        self.symbol = row['symbol']
        self.tf_name = row['tf']
        self.tf_val = getattr(mt5, f"TIMEFRAME_{row['tf']}")
        self.hold_bars = max(3, int(row['hold']))
        self.requested_sl_pips = int(row['sl'])
        self.thr = float(row['thr'])
        self.magic = 300000 + id
        
        self.last_bar_time = 0
        self.bars_held = 0
        self.model = None
        self.features = []
        self.last_retrain_time = 0
        
        # Запрашиваем у брокера реальные ограничения
        self._update_broker_constraints()
        self.retrain()

    def _update_broker_constraints(self):
        """Синхронизация с требованиями брокера"""
        info = mt5.symbol_info(self.symbol)
        if info:
            # Минимальный стоп = СтопЛевел брокера + запас
            self.min_sl_points = max(info.trade_stops_level + 50, 100)
            # Применяем реальный стоп, если запрошенный меньше разрешенного
            self.actual_sl_pips = max(self.requested_sl_pips, self.min_sl_points)
            self.point = info.point
            self.digits = info.digits
            self.volume_min = info.volume_min
            self.volume_step = info.volume_step
            
            if self.actual_sl_pips != self.requested_sl_pips:
                print(f"  [!] {self.symbol}: Брокер требует мин. SL {self.min_sl_points}. "
                      f"Адаптировано с {self.requested_sl_pips} до {self.actual_sl_pips}.")

    def retrain(self):
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.tf_val, 0, HISTORY_BARS)
            if rates is None or len(rates) < 200: return
            df = pd.DataFrame(rates)
            df['Conv_Deriv'] = gaussian_filter1d(df['close'].values, SIGMA, order=1)
            df['Target'] = np.where(df['Conv_Deriv'] > 0, 1, 0)
            df_proc, self.features = get_features(df)
            train_data = df_proc.iloc[:-LOOKAHEAD].dropna()
            
            self.model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
            self.model.fit(train_data[self.features], train_data['Target'])
            self.last_retrain_time = time.time()
            print(f"  [OK] {self.symbol} обучен/переобучен.")
        except Exception as e: 
            print(f"  [ERR] {self.symbol} retrain: {e}")

    def check_and_retrain_if_needed(self):
        """Проверка необходимости переобучения"""
        if time.time() - self.last_retrain_time > (RETRAIN_INTERVAL_HOURS * 3600):
            self._update_broker_constraints() # Обновляем лимиты брокера
            self.retrain()

    def get_current_position(self):
        pos = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        return pos[0] if pos else None

    def check_virtual_sl(self, p):
        """Проверка Виртуального Стоп-Лосса (если брокер его срезал или не поставил)"""
        if p.sl == 0:
            tick = mt5.symbol_info_tick(self.symbol)
            # Для BUY закрываем по Bid, для SELL по Ask
            current_price = tick.bid if p.type == 0 else tick.ask
            dist_points = abs(p.price_open - current_price) / self.point
            
            if dist_points >= self.actual_sl_pips:
                self.close_order(p, "VIRTUAL_SL")
                return True
        return False

    def check_signal_and_trade(self):
        """Логика входа и выхода (вызывается только на новом баре)"""
        if self.model is None: return
        
        p = self.get_current_position()
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.tf_val, 0, 100)
        if rates is None or len(rates) < 50: return
        
        curr_bar_time = rates[-1]['time']
        if curr_bar_time == self.last_bar_time: 
            return # Бар еще не закрылся/не появился новый
            
        self.last_bar_time = curr_bar_time
        
        df_live, _ = get_features(pd.DataFrame(rates))
        if df_live.empty: return
        
        prob = self.model.predict_proba(df_live.iloc[-1:][self.features])[0][1]

        if p:
            self.bars_held += 1
            reason = ""
            
            # Логика выхода
            if self.bars_held >= self.hold_bars and p.profit < 0: 
                reason = "TIMEOUT_LOSS"
            elif self.bars_held >= (self.hold_bars * 2): 
                reason = "MAX_HOLD"
            elif self.bars_held > 2:
                if (p.type == 0 and prob < (1 - self.thr - 0.03)): reason = "REVERSAL_BUY"
                elif (p.type == 1 and prob > (self.thr + 0.03)): reason = "REVERSAL_SELL"
                
            if reason: self.close_order(p, reason)
        else:
            # Логика входа
            if prob >= self.thr: self.open_order(mt5.ORDER_TYPE_BUY, prob)
            elif prob <= (1 - self.thr): self.open_order(mt5.ORDER_TYPE_SELL, prob)

    def open_order(self, order_type, prob):
        self._update_broker_constraints() # На всякий случай обновляем перед сделкой
        
        si = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        
        # Корректировка лота под требования брокера (volume_step)
        lot = max(LOT_PER_AGENT, self.volume_min)
        lot = round(lot - (lot % self.volume_step), 2) 

        price = round(tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid, self.digits)
        
        # Расчет Стоп-Лосса с учетом жестких требований брокера
        final_dist = self.actual_sl_pips * self.point
        sl = round(price - final_dist if order_type == mt5.ORDER_TYPE_BUY else price + final_dist, self.digits)

        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lot,
            "type": order_type, "price": price, "sl": sl, "magic": self.magic,
            "comment": f"V6.5 P:{prob:.2f}", "type_filling": get_filling_mode(self.symbol),
            "type_time": mt5.ORDER_TIME_GTC,
        }
        
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"!!! [{self.symbol}] OPEN {lot} (SL OK: {sl})")
        elif res.retcode in [10016, 10014]: # Invalid Stops или Invalid Volume
            print(f"  [!] {self.symbol} Ошибка {res.retcode}. Открываем БЕЗ СЛ (Virtual Guard)...")
            req['sl'] = 0.0
            res_fallback = mt5.order_send(req)
            if res_fallback.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  [OK] {self.symbol} OPENED WITHOUT SL")
            else:
                print(f"  [X] {self.symbol} FATAL REJECT: {res_fallback.comment}")
        else:
            print(f"  [!] {self.symbol} REJECTED: {res.retcode} - {res.comment}")

    def close_order(self, p, reason):
        tick = mt5.symbol_info_tick(self.symbol)
        price = round(tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask, self.digits)
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": p.volume,
            "type": mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": p.ticket, "price": price, "magic": self.magic,
            "type_filling": get_filling_mode(self.symbol), "type_time": mt5.ORDER_TIME_GTC,
        }
        if mt5.order_send(req).retcode == mt5.TRADE_RETCODE_DONE:
            print(f"X [{self.symbol}] CLOSED ({reason})")

if __name__ == "__main__":
    print("--- БОТ V6.6 (OPTIMIZED & BROKER-SYNC) ---")
    if not mt5.initialize():
        print("MT5 init failed")
        exit()
        
    configs = pd.read_csv(CONFIG_FILE)
    portfolio = [TradingAgent(i, row) for i, row in configs.iterrows()]
    
    try:
        while True:
            current_time = time.time()
            for a in portfolio:
                # 1. Редкая проверка (раз в 4 часа) - переобучение и обновление лимитов брокера
                a.check_and_retrain_if_needed()
                
                # 2. Быстрая проверка Виртуального Стопа (каждую итерацию)
                p = a.get_current_position()
                if p:
                    if a.check_virtual_sl(p):
                        continue # Если закрыли по виртуалке, идем к следующему
                
                # 3. Тяжелая проверка сигналов (внутри метода есть защита от спама - только на новом баре)
                a.check_signal_and_trade()
                
            # Спим 1 секунду. Этого достаточно для виртуального стопа, но не перегружает MT5
            time.sleep(1.0) 
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
    finally: 
        mt5.shutdown()