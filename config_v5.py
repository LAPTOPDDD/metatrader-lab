# config_v5.py
import MetaTrader5 as mt5
import numpy as np

# === Торгуемые инструменты ===
SYMBOLS = [
    "USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "XAUUSD"
]
TIMEFRAMES = [mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]
TIMEFRAME_NAMES = {
    mt5.TIMEFRAME_M30: "M30",
    mt5.TIMEFRAME_H1:  "H1",
    mt5.TIMEFRAME_H4:  "H4"
}

# === Параметры Walk-Forward ===
BARS_TOTAL = 4000
N_STEPS = 10
STEP_BARS = 20
TOTAL_OOS_BARS = N_STEPS * STEP_BARS   # 200
HOLD_BARS_RANGE = range(1, 11)

# === Диапазон множителей стоп-лосса (в ATR) ===
# Для Optuna будет использоваться непрерывный диапазон 1.0 – 3.0 с шагом 0.25
SL_MULT_MIN = 1.0
SL_MULT_MAX = 3.0
SL_MULT_STEP = 0.25

# === Пороги и фильтры ===
PROBA_BUY = 0.58
PROBA_SELL = 0.42
MIN_VAL_SHARPE = 0.85
MAX_CORRELATION = 0.65
BASE_LOT = 0.01

# === Параметры бота ===
HISTORY_BARS = 3500
CHECK_INTERVAL = 10   # секунд
RETRAIN_BARS = 20     # переобучение каждые 20 баров

# === Каталог результатов ===
OUTPUT_DIR = "walk_forward_results"