# bot_v5.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import time
import sys
import os
from datetime import datetime
from config_v5 import *

if not mt5.initialize():
    print("Ошибка подключения к MT5")
    sys.exit()

print(f"MT5 подключен. Сервер: {mt5.terminal_info().name}")

def get_features(df):
    """Расчёт фичей (строго по прошлым данным)"""
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_lag_1'] = df['MACD'].shift(1)
    df['MACD_lag_2'] = df['MACD'].shift(2)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_lag_1'] = df['RSI'].shift(1)

    df['Ret_1'] = df['close'].pct_change(1)
    df['Ret_3'] = df['close'].pct_change(3)
    df['Ret_5'] = df['close'].pct_change(5)

    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['BB_pos'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df['BB_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-9)

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_rel'] = tr.rolling(14).mean() / df['close']

    features = ['MACD_lag_1', 'MACD_lag_2', 'RSI_lag_1', 'Ret_1', 'Ret_3', 'Ret_5', 'BB_pos', 'BB_width', 'ATR_rel']
    return df, features


class StrategyAgentV5:
    def __init__(self, idx, symbol, timeframe_name, hold_bars, sl_mult, weight, total_agents):
        self.idx = idx
        self.symbol = symbol
        self.timeframe_name = timeframe_name
        tf_map = {'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4}
        self.timeframe = tf_map[timeframe_name]
        self.hold_bars = hold_bars
        self.sl_mult = sl_mult
        self.magic = 500000 + idx

        if np.isnan(weight) or weight <= 0:
            weight = 1.0 / total_agents
        self.weight = weight
        calculated_lot = round(BASE_LOT * (weight * total_agents), 2)
        self.lot = max(0.01, calculated_lot)

        self.model = None
        self.features = []
        self.last_bar_time = None
        self.bars_held = 0
        self.bars_since_retrain = 0

        self._train()

    def _train(self, retrain=False):
        """Обучение модели на HISTORY_BARS барах с отступом 5 баров от края"""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, HISTORY_BARS)
        if rates is None or len(rates) < 500:
            return

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[~df.index.dayofweek.isin([5, 6])].copy()

        # Целевая переменная (производная сглаженной цены)
        df['Conv'] = gaussian_filter1d(df['close'].values, sigma=2, mode='nearest')
        df['Conv_Diff'] = df['Conv'].diff()
        df['Target_Binary'] = (df['Conv_Diff'] > 0).astype(int)

        df, self.features = get_features(df)
        df_clean = df.dropna().copy()

        # Отступаем 5 баров, чтобы избежать краевых искажений
        train_data = df_clean.iloc[:-5].copy()
        if len(train_data) < 500:
            return

        model = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
        )
        model.fit(train_data[self.features], train_data['Target_Binary'])

        self.model = model
        self.last_bar_time = df_clean.index[-1]

        if retrain:
            self.bars_since_retrain = 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Агент {self.idx} ({self.symbol} {self.timeframe_name}) ПЕРЕОБУЧЕН")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Агент {self.idx} ({self.symbol} {self.timeframe_name}) обучен, лот={self.lot}")

    def retrain(self):
        """Принудительное переобучение (вызывается каждые RETRAIN_BARS баров)"""
        self._train(retrain=True)

    def _get_position(self):
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        return positions[0] if positions and len(positions) > 0 else None

    def _close_position(self, pos, reason=""):
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
            "comment": f"V5 Close: {reason}",
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ЗАКРЫТО #{pos.ticket} ({self.symbol}) | {reason}")
            return True
        return False

    def _get_current_atr(self):
        """Возвращает текущий ATR в пунктах"""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 20)
        if rates is None or len(rates) < 20:
            return 0.0
        df = pd.DataFrame(rates)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        return atr if not np.isnan(atr) else 0.0

    def _open_position(self, direction, proba):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return False
        point = symbol_info.point
        atr_points = self._get_current_atr()
        if atr_points == 0:
            return False
        sl_points = int(self.sl_mult * atr_points / point)  # в пунктах

        order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
        price = symbol_info.ask if direction == 'buy' else symbol_info.bid
        sl_price = price - sl_points * point if direction == 'buy' else price + sl_points * point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "magic": self.magic,
            "deviation": 20,
            "comment": f"V5 {direction.upper()} ({proba*100:.1f}%) SL={self.sl_mult}ATR",
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ОТКРЫТО {direction.upper()} ({self.symbol}) Lot={self.lot} SL={sl_points}п | Уверенность {proba*100:.1f}%")
            self.bars_held = 0
            return True
        return False

    def update(self):
        if self.model is None:
            return

        # Проверяем новый бар
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2)
        if rates is None or len(rates) < 2:
            return
        df_new = pd.DataFrame(rates)
        df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
        df_new.set_index('time', inplace=True)
        latest_bar_time = df_new.index[-1]

        if latest_bar_time == self.last_bar_time:
            return

        # === Новый бар ===
        self.last_bar_time = latest_bar_time
        self.bars_since_retrain += 1

        # 1. Проверка позиции и таймаута
        pos = self._get_position()
        if pos is not None:
            self.bars_held += 1
            if self.bars_held >= self.hold_bars:
                if self._close_position(pos, reason=f"Timeout {self.hold_bars} bars"):
                    self.bars_held = 0
                    # не выходим, даём модели возможность открыть новую на этом же баре

        # 2. Прогноз
        rates_hist = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 100)
        if rates_hist is None:
            return
        df_h = pd.DataFrame(rates_hist)
        df_h['time'] = pd.to_datetime(df_h['time'], unit='s')
        df_h.set_index('time', inplace=True)
        df_h = df_h[~df_h.index.dayofweek.isin([5, 6])].copy()
        df_h, _ = get_features(df_h)
        df_h = df_h.dropna().copy()
        if df_h.empty:
            return

        last_row = df_h.iloc[-1:][self.features]
        prob_buy = self.model.predict_proba(last_row)[0][1]

        desired_dir = None
        if prob_buy >= PROBA_BUY:
            desired_dir = 'buy'
        elif prob_buy <= PROBA_SELL:
            desired_dir = 'sell'

        # 3. Управление позицией
        pos = self._get_position()   # перечитываем, т.к. могли закрыть по таймауту
        if pos is not None:
            current_dir = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
            if desired_dir is not None and current_dir != desired_dir:
                if self._close_position(pos, reason="Signal Reversal"):
                    self._open_position(desired_dir, prob_buy if desired_dir == 'buy' else 1 - prob_buy)
            # если направления совпадают – ничего не делаем
        else:
            if desired_dir is not None:
                self._open_position(desired_dir, prob_buy if desired_dir == 'buy' else 1 - prob_buy)


# ================== ЗАПУСК БОТА ==================
csv_path = os.path.join(OUTPUT_DIR, 'walk_forward_strategies_v5.csv')
if not os.path.exists(csv_path):
    print(f"Файл {csv_path} не найден! Запустите train_v5_optuna.py")
    mt5.shutdown()
    sys.exit()

df_strats = pd.read_csv(csv_path)
agents = []
total_agents = len(df_strats)

print("\n=== Инициализация агентов V5 ===")
for idx, row in df_strats.iterrows():
    agent = StrategyAgentV5(
        idx,
        row['symbol'],
        row['timeframe'],
        int(row['hold_bars']),
        float(row['sl_mult']),
        float(row['weight']),
        total_agents
    )
    if agent.model is not None:
        agents.append(agent)

print(f"\nЗапущено агентов: {len(agents)} из {total_agents}")
print("Бот V5 работает. Переобучение каждые 20 баров для каждого агента.\n")

try:
    while True:
        for agent in agents:
            agent.update()
            # Переобучение при достижении RETRAIN_BARS
            if agent.bars_since_retrain >= RETRAIN_BARS:
                agent.retrain()
        time.sleep(CHECK_INTERVAL)
except KeyboardInterrupt:
    print("\nБот остановлен пользователем.")
finally:
    mt5.shutdown()