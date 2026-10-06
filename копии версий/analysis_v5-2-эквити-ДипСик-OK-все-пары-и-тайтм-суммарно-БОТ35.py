import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import time
import sys
import os

# ------------------- НАСТРОЙКИ -------------------
TIMEFRAME_NAMES = {
    mt5.TIMEFRAME_M5: "M5",
    mt5.TIMEFRAME_M15: "M15",
    mt5.TIMEFRAME_M30: "M30",
    mt5.TIMEFRAME_H1: "H1",
    mt5.TIMEFRAME_H4: "H4",
}
LOT = 0.01
CHECK_INTERVAL = 10  # секунд
HISTORY_BARS = 500

if not mt5.initialize():
    print("Ошибка инициализации MT5")
    sys.exit()
print("MT5 подключён.")

# Загрузка стратегий (из CSV или встроенного списка)
def load_strategies():
    csv_path = "ensemble_results/top_35_strategies.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        strategies = df[['symbol', 'timeframe', 'hold_bars']].to_dict('records')
        print(f"Загружено {len(strategies)} стратегий из CSV.")
        return strategies
    else:
        print("CSV не найден, используем встроенный пример (замените на свои топ-35).")
        # Здесь укажите свои топ-35, если нет CSV
        sample = [
            {'symbol': 'USDJPY', 'timeframe': 'H4', 'hold_bars': 4},
            {'symbol': 'USDJPY', 'timeframe': 'H4', 'hold_bars': 9},
            # ... добавьте все 35
        ]
        return sample

strategies_config = load_strategies()
tf_map = {'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
          'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
          'H4': mt5.TIMEFRAME_H4}

# ------------------- КЛАСС СТРАТЕГИИ С УДЕРЖАНИЕМ -------------------
class Strategy:
    def __init__(self, idx, symbol, timeframe_name, hold_bars):
        self.idx = idx
        self.symbol = symbol
        self.timeframe_name = timeframe_name
        self.timeframe = tf_map[timeframe_name]
        self.hold_bars = hold_bars
        self.magic = 100000 + idx
        self.model = None
        self.features = [f'MACD_lag_{i}' for i in range(1, 6)]
        self.last_bar_time = None
        self.current_position_type = None  # 'buy' or 'sell' or None
        self.entry_time = None             # время открытия позиции
        self.bars_held = 0                 # сколько баров удерживаем

        self._train_model()

    def _train_model(self):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, HISTORY_BARS)
        if rates is None or len(rates) < 100:
            print(f"Недостаточно данных для {self.symbol} {self.timeframe_name}")
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
        if len(df_clean) < 50:
            print(f"Мало данных для {self.symbol} {self.timeframe_name}")
            return

        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)
        model.fit(df_clean[self.features], df_clean['Target_Binary'])
        self.model = model
        self.last_bar_time = df_clean.index[-1]
        print(f"Стратегия {self.idx}: {self.symbol} {self.timeframe_name} обучена (hold={self.hold_bars})")

    def _get_current_position(self):
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if positions and len(positions) > 0:
            return positions[0]
        return None

    def _close_position(self, pos):
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
            "comment": "Close by signal/timeout",
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Закрыта позиция {pos.ticket} для {self.symbol} {self.timeframe_name} (magic={self.magic})")
            return True
        else:
            print(f"Ошибка закрытия {pos.ticket}: {result.comment}")
            return False

    def _open_position(self, direction):
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
            "comment": f"Signal {direction}",
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Открыта {direction} позиция для {self.symbol} {self.timeframe_name} (magic={self.magic})")
            self.entry_time = self.last_bar_time
            self.bars_held = 0
            return True
        else:
            print(f"Ошибка открытия {direction} для {self.symbol} {self.timeframe_name}: {result.comment}")
            return False

    def check_and_trade(self):
        if self.model is None:
            return

        # Получаем последние 2 бара для проверки нового
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2)
        if rates is None or len(rates) < 2:
            return
        df_new = pd.DataFrame(rates)
        df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
        df_new.set_index('time', inplace=True)
        latest_bar_time = df_new.index[-1]

        if latest_bar_time == self.last_bar_time:
            return  # нового бара нет

        # Новый бар – обновляем last_bar_time
        self.last_bar_time = latest_bar_time

        # Если есть открытая позиция – увеличиваем счётчик удержания
        pos = self._get_current_position()
        if pos is not None:
            self.bars_held += 1
            # Проверяем, не истекло ли время удержания
            if self.bars_held >= self.hold_bars:
                # Закрываем позицию по таймауту
                if self._close_position(pos):
                    self.current_position_type = None
                    self.entry_time = None
                    self.bars_held = 0
                # После закрытия по таймауту не открываем новую, ждём следующего сигнала
                # (но можно и сразу открыть, если сигнал изменился – обработаем дальше)

        # Пересчитываем индикаторы на последних 100 барах для нового прогноза
        rates_hist = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 100)
        if rates_hist is None:
            return
        df_hist = pd.DataFrame(rates_hist)
        df_hist['time'] = pd.to_datetime(df_hist['time'], unit='s')
        df_hist.set_index('time', inplace=True)
        df_hist = df_hist[~df_hist.index.dayofweek.isin([5, 6])].copy()

        df_hist['Conv'] = gaussian_filter1d(df_hist['close'].values, sigma=2, mode='nearest')
        df_hist['Conv_Diff'] = df_hist['Conv'].diff()
        df_hist['MACD'] = df_hist['close'].ewm(span=12, adjust=False).mean() - df_hist['close'].ewm(span=26, adjust=False).mean()
        for i in range(1, 6):
            df_hist[f'MACD_lag_{i}'] = df_hist['MACD'].shift(i)
        df_hist = df_hist.dropna().copy()
        if df_hist.empty:
            return

        last_row = df_hist.iloc[-1:]
        pred = self.model.predict(last_row[self.features])[0]
        desired_direction = 'buy' if pred == 1 else 'sell'

        # Проверяем текущую позицию (возможно, уже закрылась по таймауту)
        pos = self._get_current_position()
        if pos is not None:
            current_dir = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
            if current_dir == desired_direction:
                # Направление совпадает – ничего не делаем
                return
            else:
                # Сигнал изменился – закрываем и открываем новую
                if self._close_position(pos):
                    self.current_position_type = None
                    self.entry_time = None
                    self.bars_held = 0
                    # Открываем новую позицию
                    if self._open_position(desired_direction):
                        self.current_position_type = desired_direction
        else:
            # Нет позиции – открываем, если сигнал есть (а он всегда есть)
            if self._open_position(desired_direction):
                self.current_position_type = desired_direction

# ------------------- СОЗДАНИЕ И ЗАПУСК -------------------
strategies = []
for idx, cfg in enumerate(strategies_config):
    strat = Strategy(idx, cfg['symbol'], cfg['timeframe'], cfg['hold_bars'])
    if strat.model is not None:
        strategies.append(strat)

print(f"Активных стратегий: {len(strategies)}")

try:
    while True:
        for strat in strategies:
            strat.check_and_trade()
        time.sleep(CHECK_INTERVAL)
except KeyboardInterrupt:
    print("Робот остановлен.")
finally:
    mt5.shutdown()