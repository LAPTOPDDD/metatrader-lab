import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import time
import sys
import os
from datetime import datetime

# ================== НАСТРОЙКИ BOTA V5 ==================
BASE_LOT = 0.01                        # Базовый лот
CHECK_INTERVAL = 10                    # Пауза опроса баров (сек)
HISTORY_BARS = 3500                    # Свежая история для обучения
PROBA_BUY = 0.58                       # Порог уверенности (58%)
PROBA_SELL = 0.42
CSV_PATH = "walk_forward_results/walk_forward_strategies_v5.csv"

TF_MAP = {
    'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4
}

if not mt5.initialize():
    print(" Ошибка подключения к MT5")
    sys.exit()

print(f" MT5 Успешно подключен. Сервер: {mt5.terminal_info().name}")

def get_features(df):
    """Единая функция расчета каузальных фичей"""
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
    def __init__(self, idx, symbol, timeframe_name, hold_bars, weight, total_agents):
        self.idx = idx
        self.symbol = symbol
        self.timeframe_name = timeframe_name
        self.timeframe = TF_MAP[timeframe_name]
        self.hold_bars = hold_bars
        self.magic = 500000 + idx
        
        # Страховка от NaN в весах
        if np.isnan(weight) or weight <= 0:
            weight = 1.0 / total_agents
            
        self.weight = weight
        calculated_lot = round(BASE_LOT * (weight * total_agents), 2)
        self.lot = max(0.01, calculated_lot)
        
        self.model = None
        self.features = []
        self.last_bar_time = None
        self.bars_held = 0

        self._train()

    def _train(self):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, HISTORY_BARS)
        if rates is None or len(rates) < 500:
            return

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[~df.index.dayofweek.isin([5, 6])].copy()

        df['Conv'] = gaussian_filter1d(df['close'].values, sigma=2, mode='nearest')
        df['Conv_Diff'] = df['Conv'].diff()
        df['Target_Binary'] = (df['Conv_Diff'] > 0).astype(int)

        df, self.features = get_features(df)
        df_clean = df.dropna().copy()

        model = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
        )
        model.fit(df_clean[self.features], df_clean['Target_Binary'])
        
        self.model = model
        self.last_bar_time = df_clean.index[-1]
        print(f" Агент V5 #{self.idx:02d} [{self.symbol} {self.timeframe_name} Hold={self.hold_bars}] Lot={self.lot} (W={self.weight:.3f}) готов.")

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
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ZAKRYTO #{pos.ticket} ({self.symbol} {self.timeframe_name}) | {reason}")
            return True
        return False

    def _open_position(self, direction, proba):
        order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if direction == 'buy' else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot,
            "type": order_type,
            "price": price,
            "magic": self.magic,
            "deviation": 20,
            "comment": f"V5 {direction.upper()} ({proba*100:.1f}%)",
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] OTKRYTO {direction.upper()} ({self.symbol} {self.timeframe_name}) Lot={self.lot} Уверенность: {proba*100:.1f}%")
            self.bars_held = 0
            return True
        return False

    def update(self):
        if self.model is None:
            return

        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2)
        if rates is None or len(rates) < 2:
            return
            
        df_new = pd.DataFrame(rates)
        df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
        df_new.set_index('time', inplace=True)
        latest_bar_time = df_new.index[-1]

        if latest_bar_time == self.last_bar_time:
            return

        self.last_bar_time = latest_bar_time

        # 1. Позиция по таймауту
        pos = self._get_position()
        if pos is not None:
            self.bars_held += 1
            if self.bars_held >= self.hold_bars:
                if self._close_position(pos, reason=f"Timeout {self.hold_bars} bars"):
                    self.bars_held = 0
                    return # ИСПРАВЛЕНО: Ждем следующего закрытого бара!

        # 2. Расчет признаков для нового прогноза
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

        if pos is not None:
            current_dir = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
            if desired_dir is not None and current_dir != desired_dir:
                if self._close_position(pos, reason="Signal Reversal"):
                    self._open_position(desired_dir, prob_buy if desired_dir == 'buy' else 1 - prob_buy)
        else:
            if desired_dir is not None:
                self._open_position(desired_dir, prob_buy if desired_dir == 'buy' else 1 - prob_buy)

# ================== ЗАПУСК АНСАМБЛЯ V5 ==================
if not os.path.exists(CSV_PATH):
    print(f" Файл {CSV_PATH} не найден! Запустите build_ensemble_v5.py")
    mt5.shutdown()
    sys.exit()

df_strats = pd.read_csv(CSV_PATH)
agents = []
total_agents = len(df_strats)

print("\n=== Инициализация Walk-Forward Ансамбля V5 ===")
for idx, row in df_strats.iterrows():
    agent = StrategyAgentV5(
        idx, row['symbol'], row['timeframe'], 
        int(row['hold_bars']), float(row['weight']), total_agents
    )
    if agent.model is not None:
        agents.append(agent)

print(f"\n Запущено Walk-Forward агентов V5: {len(agents)} из {total_agents}")
print(" Бот V5 в режиме ожидания уверенных сигналов...\n")

try:
    while True:
        for agent in agents:
            agent.update()
        time.sleep(CHECK_INTERVAL)
except KeyboardInterrupt:
    print("\n Бот V5 остановлен.")
finally:
    mt5.shutdown()