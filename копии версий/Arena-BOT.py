#!/usr/bin/env python3
"""
=============================================================================
TRADING BOT V5  —  Walk-Forward с Переобучением
=============================================================================
Рабочий бот, использующий ансамбль агентов LightGBM.
Каждый агент переобучается каждые 20 баров своего таймфрейма.
Встроен адаптивный стоп-лосс (ATR × sl_mult).

Требования:
  - MetaTrader 5 (терминал + Python API)
  - walk_forward_strategies_v5.csv  (выход тренировочного скрипта)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from scipy.ndimage import gaussian_filter1d
from lightgbm import LGBMClassifier
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 1. КОНСТАНТЫ
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH       = "walk_forward_strategies_v5.csv"
BASE_LOT       = 0.01
HISTORY_BARS   = 3500
EDGE_TRIM      = 5          # Обрезка правого края при обучении
RETRAIN_EVERY  = 20         # Переобучение каждые N баров
CHECK_INTERVAL = 10         # Пауза между итерациями (сек)

# Пороги сигналов
PROB_BUY  = 0.58
PROB_SELL = 0.42

TIMEFRAMES = {
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
}

LGB_PARAMS = dict(
    n_estimators=120,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=42,
)

FEATURE_COLS = [
    "MACD_lag_1", "MACD_lag_2",
    "RSI_lag_1",
    "Ret_1", "Ret_3", "Ret_5",
    "BB_pos", "BB_width",
    "ATR_rel",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────
def log(msg: str):
    """Вывод в консоль с меткой времени."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def load_bars(symbol: str, tf_name: str, n_bars: int) -> pd.DataFrame:
    """Загрузить OHLCV и удалить выходные."""
    tf_mt5 = TIMEFRAMES[tf_name]
    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise ValueError(f"Нет данных для {symbol} {tf_name}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[~df["time"].dt.weekday.isin([5, 6])].reset_index(drop=True)
    return df


def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать каузальные (лаговые) признаки."""
    close = df["close"].copy()
    high  = df["high"].copy()
    low   = df["low"].copy()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df = df.copy()
    df["MACD_lag_1"] = macd.shift(1)
    df["MACD_lag_2"] = macd.shift(2)

    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)
    df["RSI_lag_1"] = rsi.shift(1)

    df["Ret_1"] = close.pct_change(1)
    df["Ret_3"] = close.pct_change(3)
    df["Ret_5"] = close.pct_change(5)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BB_pos"]   = (close - lower) / (upper - lower).replace(0, np.nan)
    df["BB_width"] = (upper - lower) / sma20.replace(0, np.nan)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df["ATR_rel"] = atr14 / close
    df["ATR_abs"] = atr14

    return df


def calc_target(df: pd.DataFrame) -> pd.Series:
    """Целевая: знак производной гауссово-сглаженной цены."""
    conv = gaussian_filter1d(df["close"].values, sigma=2, mode="nearest")
    conv_diff = pd.Series(conv, index=df.index).diff()
    return (conv_diff > 0).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 3. КЛАСС АГЕНТА
# ─────────────────────────────────────────────────────────────────────────────
class StrategyAgentV5:
    """Один агент ансамбля — один символ + таймфрейм + параметры."""

    def __init__(
        self,
        idx: int,
        symbol: str,
        timeframe: str,
        hold_bars: int,
        sl_mult: float,
        weight: float,
        total_agents: int,
    ):
        self.idx       = idx
        self.symbol    = symbol
        self.timeframe = timeframe
        self.hold_bars = hold_bars
        self.sl_mult   = sl_mult
        self.weight    = weight
        self.magic      = 500_000 + idx
        self.lot        = round(BASE_LOT * weight * total_agents, 2)
        self.lot        = max(self.lot, 0.01)

        self.model              = None
        self.last_bar_time      = None
        self.bars_since_retrain = 0
        self.bars_held          = 0

        log(f"Agent #{idx} created: {symbol} {timeframe} "
            f"hold={hold_bars} sl={sl_mult} w={weight:.4f} lot={self.lot}")

        # Первичное обучение
        try:
            self._train()
        except Exception as e:
            log(f"Agent #{idx} TRAIN FAILED: {e}")

    # ─── Обучение ────────────────────────────────────────────────────────
    def _train(self):
        """Первичное (или повторное) обучение модели."""
        df = load_bars(self.symbol, self.timeframe, HISTORY_BARS)
        df = calc_features(df)
        df["target"] = calc_target(df)
        df.dropna(subset=FEATURE_COLS + ["target"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Обрезать EDGE_TRIM баров с правого края
        df = df.iloc[:-EDGE_TRIM]

        X = df[FEATURE_COLS]
        y = df["target"]

        self.model = LGBMClassifier(**LGB_PARAMS)
        self.model.fit(X, y)

        self.last_bar_time = df["time"].iloc[-1]
        log(f"Agent #{self.idx} TRAINED on {len(df)} bars  "
            f"(last bar: {self.last_bar_time})")

    def retrain(self):
        """Переобучение без закрытия позиций."""
        log(f"Agent #{self.idx} RETRAINING …")
        try:
            self._train()
            self.bars_since_retrain = 0
        except Exception as e:
            log(f"Agent #{self.idx} RETRAIN FAILED: {e}")

    # ─── Торговые операции ───────────────────────────────────────────────
    def _get_position(self):
        """Получить открытую позицию по magic."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for p in positions:
                if p.magic == self.magic:
                    return p
        return None

    def _open_position(self, direction: int, prob: float):
        """Открыть позицию с ATR-стопом."""
        info = mt5.symbol_info_tick(self.symbol)
        if info is None:
            return

        # Текущий ATR
        df_short = load_bars(self.symbol, self.timeframe, 100)
        df_short = calc_features(df_short)
        atr = df_short["ATR_abs"].iloc[-1]
        if np.isnan(atr):
            atr = 0

        point = mt5.symbol_info(self.symbol).point

        if direction == 1:
            price  = info.ask
            sl     = round(price - self.sl_mult * atr, 5)
            order  = mt5.ORDER_TYPE_BUY
            label  = "BUY"
        else:
            price  = info.bid
            sl     = round(price + self.sl_mult * atr, 5)
            order  = mt5.ORDER_TYPE_SELL
            label  = "SELL"

        comment = f"V5 {label} ({prob*100:.1f}%) SL={self.sl_mult}ATR"

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    self.symbol,
            "volume":    self.lot,
            "type":      order,
            "price":     price,
            "sl":        sl,
            "deviation": 20,
            "magic":     self.magic,
            "comment":   comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log(f"Agent #{self.idx}  OPENED {label}  {self.symbol}  "
                f"lot={self.lot}  sl={sl}  [{comment}]")
            self.bars_held = 0
        else:
            err = result.retcode if result else "None"
            log(f"Agent #{self.idx}  OPEN FAILED  {self.symbol}  retcode={err}")

    def _close_position(self, position, reason: str):
        """Закрыть позицию."""
        info = mt5.symbol_info_tick(self.symbol)
        if info is None:
            return

        if position.type == mt5.ORDER_TYPE_BUY:
            price = info.bid
            order = mt5.ORDER_TYPE_SELL
        else:
            price = info.ask
            order = mt5.ORDER_TYPE_BUY

        comment = f"V5 Close: {reason}"

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    self.symbol,
            "volume":    position.volume,
            "type":      order,
            "price":     price,
            "position":  position.ticket,
            "deviation": 20,
            "magic":     self.magic,
            "comment":   comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log(f"Agent #{self.idx}  CLOSED  {self.symbol}  [{comment}]")
        else:
            err = result.retcode if result else "None"
            log(f"Agent #{self.idx}  CLOSE FAILED  {self.symbol}  retcode={err}")

    # ─── Основной update ─────────────────────────────────────────────────
    def update(self):
        """Вызывается на каждой итерации цикла."""
        if self.model is None:
            return

        # 1. Проверить новый бар
        try:
            df = load_bars(self.symbol, self.timeframe, 100)
        except ValueError:
            return

        current_bar_time = df["time"].iloc[-1]
        if self.last_bar_time is not None and current_bar_time <= self.last_bar_time:
            return  # нового бара нет

        self.last_bar_time = current_bar_time
        self.bars_since_retrain += 1

        # 2. Обработать текущую позицию
        pos = self._get_position()
        if pos is not None:
            self.bars_held += 1
            if self.bars_held >= self.hold_bars:
                self._close_position(pos, "Timeout")
                pos = None  # позиция закрыта, можно открыть новую

        # 3. Прогноз
        df = calc_features(df)
        df.dropna(subset=FEATURE_COLS, inplace=True)
        if df.empty:
            return

        X_last = df[FEATURE_COLS].iloc[[-1]]
        prob = self.model.predict_proba(X_last)[0, 1]

        # Желаемое направление
        if prob >= PROB_BUY:
            desired = 1
        elif prob <= PROB_SELL:
            desired = -1
        else:
            desired = 0

        # 4. Управление позицией
        pos = self._get_position()  # перечитать (могла быть закрыта выше)

        if pos is not None:
            current_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            if desired != 0 and desired != current_dir:
                self._close_position(pos, "Signal Reversal")
                self._open_position(desired, prob)
        else:
            if desired != 0:
                self._open_position(desired, prob)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ГЛАВНЫЙ ЦИКЛ
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    log(f"MT5 connected — build {mt5.version()}")

    # Загрузить CSV
    strategies = pd.read_csv(CSV_PATH)
    total_agents = len(strategies)
    log(f"Loaded {total_agents} strategies from {CSV_PATH}")

    # Создать агентов
    agents: list[StrategyAgentV5] = []
    for i, row in strategies.iterrows():
        agent = StrategyAgentV5(
            idx=i,
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            hold_bars=int(row["hold_bars"]),
            sl_mult=float(row["sl_mult"]),
            weight=float(row["weight"]),
            total_agents=total_agents,
        )
        agents.append(agent)

    log(f"All {len(agents)} agents initialised.  Starting main loop …")

    # Бесконечный цикл
    try:
        while True:
            for agent in agents:
                try:
                    agent.update()
                except Exception as e:
                    log(f"Agent #{agent.idx} update ERROR: {e}")

            # Переобучение при необходимости
            for agent in agents:
                if agent.bars_since_retrain >= RETRAIN_EVERY:
                    try:
                        agent.retrain()
                    except Exception as e:
                        log(f"Agent #{agent.idx} retrain ERROR: {e}")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        log("Bot stopped by user.")
    finally:
        mt5.shutdown()
        log("MT5 shutdown. Bye.")


if __name__ == "__main__":
    main()
