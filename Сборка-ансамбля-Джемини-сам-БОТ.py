import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import time
import sys
import os
from datetime import datetime

# ================== НАСТРОЙКИ ==================
LOT = 0.01                     # Объём сделки для каждого агента
CHECK_INTERVAL = 10            # Пауза между проверками баров (сек)
HISTORY_BARS = 3500            # Глубина истории для обучения
CSV_PATH = "ensemble_results/top_35_strategies.csv"

TF_MAP = {
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4
}

if not mt5.initialize():
    print(" Ошибка подключения к MetaTrader 5")
    sys.exit()

print(f" MT5 Успешно подключен. Сервер: {mt5.terminal_info().name}")

# ================== КЛАСС ТОРГОВОГО АГЕНТА ==================
class StrategyAgent:
    def __init__(self, idx, symbol, timeframe_name, hold_bars):
        self.idx = idx
        self.symbol = symbol
        self.timeframe_name = timeframe_name
        self.timeframe = TF_MAP[timeframe_name]
        self.hold_bars = hold_bars
        self.magic = 200000 + idx
        self.features = [f'MACD_lag_{i}' for i in range(1, 6)]
        
        self.model = None
        self.last_bar_time = None
        self.bars_held = 0

        self._train()

    def _train(self):
        """Обучение модели агента на свежей истории"""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, HISTORY_BARS)
        if rates is None or len(rates) < 500:
            print(f" Недостаточно данных для {self.symbol} {self.timeframe_name}")
            return

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[~df.index.dayofweek.isin([5, 6])].copy()

        df['Conv'] = gaussian_filter1d(df['close'].values, sigma=2, mode='nearest')
        df['Conv_Diff'] = df['Conv'].diff()
        df['Target_Binary'] = (df['Conv_Diff'] > 0).astype(int)

        df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        for i in range(1, 6):
            df[f'MACD_lag_{i}'] = df['MACD'].shift(i)

        df_clean = df.dropna().copy()
        
        model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=3,
            random_state=42, verbosity=-1
        )
        model.fit(df_clean[self.features], df_clean['Target_Binary'])
        
        self.model = model
        self.last_bar_time = df_clean.index[-1]
        print(f" Стратегия #{self.idx:02d} [{self.symbol} {self.timeframe_name} Hold={self.hold_bars}] обучена.")

    def _get_position(self):
        """Проверка открытой позиции данного агента по Magic Number"""
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if positions and len(positions) > 0:
            return positions[0]
        return None

    def _close_position(self, pos, reason=""):
        """Закрытие существующей позиции"""
        order_type = mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "magic": self.magic,
            "deviation": 20,
            "comment": f"Close: {reason}",
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ZAKRYTO #{pos.ticket} ({self.symbol} {self.timeframe_name}) | Причина: {reason}")
            return True
        else:
            print(f" Ошибка закрытия #{pos.ticket}: {res.comment}")
            return False

    def _open_position(self, direction):
        """Открытие новой позиции (BUY или SELL)"""
        order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if direction == 'buy' else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": LOT,
            "type": order_type,
            "price": price,
            "magic": self.magic,
            "deviation": 20,
            "comment": f"Signal {direction.upper()}",
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] OTKRYTO {direction.upper()} ({self.symbol} {self.timeframe_name}) Magic={self.magic}")
            self.bars_held = 0
            return True
        else:
            print(f" Ошибка открытия {direction}: {res.comment}")
            return False

    def update(self):
        if self.model is None:
            return

        # 1. Проверяем появление нового закрытого бара
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2)
        if rates is None or len(rates) < 2:
            return
        
        df_new = pd.DataFrame(rates)
        df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
        df_new.set_index('time', inplace=True)
        latest_bar_time = df_new.index[-1]

        if latest_bar_time == self.last_bar_time:
            return  # Новый бар еще не закрылся

        self.last_bar_time = latest_bar_time

        # 2. Позиция существует -> увеличиваем счетчик и проверяем таймаут
        pos = self._get_position()
        if pos is not None:
            self.bars_held += 1
            if self.bars_held >= self.hold_bars:
                self._close_position(pos, reason=f"Timeout {self.hold_bars} bars")
                pos = None  # Позиция закрыта

        # 3. Расчет свежих фичей на истории
        rates_hist = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 100)
        if rates_hist is None:
            return
            
        df_h = pd.DataFrame(rates_hist)
        df_h['time'] = pd.to_datetime(df_h['time'], unit='s')
        df_h.set_index('time', inplace=True)
        df_h = df_h[~df_h.index.dayofweek.isin([5, 6])].copy()

        df_h['MACD'] = df_h['close'].ewm(span=12, adjust=False).mean() - df_h['close'].ewm(span=26, adjust=False).mean()
        for i in range(1, 6):
            df_h[f'MACD_lag_{i}'] = df_h['MACD'].shift(i)
            
        df_h = df_h.dropna().copy()
        if df_h.empty:
            return

        # 4. Прогноз модели
        last_features = df_h.iloc[-1:][self.features]
        pred = self.model.predict(last_features)[0]
        desired_dir = 'buy' if pred == 1 else 'sell'

        # 5. Управление сделкой
        if pos is not None:
            current_dir = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
            if current_dir != desired_dir:
                # Сигнал изменился на противоположный -> разворот
                if self._close_position(pos, reason="Signal Reversal"):
                    self._open_position(desired_dir)
        else:
            # Открываем новую позицию по сигналу
            self._open_position(desired_dir)

# ================== ИНИЦИАЛИЗАЦИЯ И ЗАПУСК АНСАМБЛЯ ==================
if not os.path.exists(CSV_PATH):
    print(f" Файл {CSV_PATH} не найден! Сначала запустите build_ensemble.py")
    mt5.shutdown()
    sys.exit()

df_strategies = pd.read_csv(CSV_PATH)
agents = []

print("\n=== Инициализация и обучение агентов ансамбля ===")
for idx, row in df_strategies.iterrows():
    agent = StrategyAgent(idx, row['symbol'], row['timeframe'], int(row['hold_bars']))
    if agent.model is not None:
        agents.append(agent)

print(f"\n Успешно запущено агентов: {len(agents)} из {len(df_strategies)}")
print(" Бот перешел в режим ожидания новых баров. Нажмите Ctrl+C для остановки.\n")

try:
    while True:
        for agent in agents:
            agent.update()
        time.sleep(CHECK_INTERVAL)
except KeyboardInterrupt:
    print("\n Бот остановлен пользователем.")
finally:
    mt5.shutdown()