#!/usr/bin/env python3
"""
=============================================================================
WALK-FORWARD TRAINING SCRIPT V5
=============================================================================
Тренировочный скрипт для алгоритмического торгового бота V5.
Выполняет Walk-Forward оптимизацию с переобучением, подбирает ансамбль
стратегий на основе LightGBM, оптимизирует стоп-лосс (ATR-множитель).

Выход:
  - walk_forward_strategies_v5.csv  (отобранные стратегии с весами)
  - walk_forward_ensemble.png       (кривая совокупной доходности)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from scipy.ndimage import gaussian_filter1d
from lightgbm import LGBMClassifier
from itertools import product
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. КОНСТАНТЫ
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = [
    "USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "XAUUSD"
]

TIMEFRAMES = {
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
}

BARS_TOTAL    = 4000      # Количество баров для загрузки
N_STEPS       = 10        # Количество Walk-Forward окон
STEP_BARS     = 20        # Размер одного окна (баров)
HOLD_RANGE    = range(1, 11)                              # hold_bars: 1..10
SL_MULT_RANGE = np.arange(1.0, 3.25, 0.25)               # sl_mult: 1.0..3.0

# Параметры модели LightGBM
LGB_PARAMS = dict(
    n_estimators=120,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=42,
)

# Пороги сигналов
PROB_BUY  = 0.58
PROB_SELL = 0.42

# Sharpe / отбор
SHARPE_THRESHOLD = 0.85
CORR_THRESHOLD   = 0.65
ANNUAL_FACTOR    = np.sqrt(252)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────
def init_mt5():
    """Инициализация терминала MT5."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    print(f"[INFO] MT5 initialised  – build {mt5.version()}")


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
    """
    Рассчитать каузальные (лаговые) признаки.
    Возвращает копию df с добавленными столбцами фичей.
    """
    close = df["close"].copy()
    high  = df["high"].copy()
    low   = df["low"].copy()

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df = df.copy()
    df["MACD_lag_1"] = macd.shift(1)
    df["MACD_lag_2"] = macd.shift(2)

    # --- RSI 14 ---
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=13, adjust=False).mean()
    avg_l  = loss.ewm(com=13, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    rsi    = 100 - 100 / (1 + rs)
    df["RSI_lag_1"] = rsi.shift(1)

    # --- Returns ---
    df["Ret_1"] = close.pct_change(1)
    df["Ret_3"] = close.pct_change(3)
    df["Ret_5"] = close.pct_change(5)

    # --- Bollinger Bands (20) ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BB_pos"]   = (close - lower) / (upper - lower).replace(0, np.nan)
    df["BB_width"] = (upper - lower) / sma20.replace(0, np.nan)

    # --- ATR rel ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df["ATR_rel"] = atr14 / close
    df["ATR_abs"] = atr14             # для стоп-лосса

    return df


FEATURE_COLS = [
    "MACD_lag_1", "MACD_lag_2",
    "RSI_lag_1",
    "Ret_1", "Ret_3", "Ret_5",
    "BB_pos", "BB_width",
    "ATR_rel",
]


def calc_target(df: pd.DataFrame) -> pd.Series:
    """Целевая переменная: знак производной гауссово-сглаженной цены."""
    conv = gaussian_filter1d(df["close"].values, sigma=2, mode="nearest")
    conv_diff = pd.Series(conv, index=df.index).diff()
    target = (conv_diff > 0).astype(int)
    return target


# ─────────────────────────────────────────────────────────────────────────────
# 3. СИМУЛЯЦИЯ ТОРГОВЛИ НА ОКНЕ
# ─────────────────────────────────────────────────────────────────────────────
def simulate_window(
    probs: np.ndarray,
    closes: np.ndarray,
    atr_vals: np.ndarray,
    hold_bars: int,
    sl_mult: float,
) -> np.ndarray:
    """
    Побаровая симуляция торговли на out-of-sample окне.
    Возвращает массив PnL длиной len(probs).
    """
    n = len(probs)
    pnl = np.zeros(n)
    pos = 0        # +1 = long, -1 = short, 0 = flat
    entry_px = 0.0
    entry_sl = 0.0
    bars_held = 0

    for i in range(n):
        p = probs[i]
        c = closes[i]
        atr = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0

        # --- Определить желаемый сигнал ---
        if p >= PROB_BUY:
            sig = 1
        elif p <= PROB_SELL:
            sig = -1
        else:
            sig = 0

        # --- Стоп-лосс проверка ---
        if pos != 0:
            sl_hit = False
            if pos == 1 and c <= entry_sl:
                sl_hit = True
            elif pos == -1 and c >= entry_sl:
                sl_hit = True

            if sl_hit:
                pnl[i] = pos * (c - entry_px)
                pos = 0
                bars_held = 0
                continue

        # --- Таймаут ---
        if pos != 0:
            bars_held += 1
            if bars_held >= hold_bars:
                pnl[i] = pos * (c - entry_px)
                pos = 0
                bars_held = 0
                # Дать возможность открыть новую позицию ниже

        # --- Разворот ---
        if pos != 0 and sig != 0 and sig != pos:
            pnl[i] += pos * (c - entry_px)
            pos = 0
            bars_held = 0

        # --- Открытие ---
        if pos == 0 and sig != 0:
            pos = sig
            entry_px = c
            if sig == 1:
                entry_sl = c - sl_mult * atr
            else:
                entry_sl = c + sl_mult * atr
            bars_held = 0

    return pnl


# ─────────────────────────────────────────────────────────────────────────────
# 4. WALK-FORWARD ЦИКЛ
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward(symbol: str, tf_name: str, hold_bars: int, sl_mult: float):
    """
    Провести Walk-Forward для одной комбинации параметров.
    Возвращает dict с метриками или None при ошибке.
    """
    try:
        df = load_bars(symbol, tf_name, BARS_TOTAL)
    except ValueError:
        return None

    df = calc_features(df)
    df["target"] = calc_target(df)
    df.dropna(subset=FEATURE_COLS + ["target"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 500:
        return None

    total_oos = N_STEPS * STEP_BARS          # 200
    all_pnl = []

    for s in range(N_STEPS):
        test_start = len(df) - (N_STEPS - s) * STEP_BARS
        test_end   = test_start + STEP_BARS

        if test_start < 200:
            return None

        train = df.iloc[:test_start]
        test  = df.iloc[test_start:test_end]

        X_train = train[FEATURE_COLS]
        y_train = train["target"]
        X_test  = test[FEATURE_COLS]

        model = LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train, y_train)

        probs    = model.predict_proba(X_test)[:, 1]
        closes   = test["close"].values
        atr_vals = test["ATR_abs"].values

        pnl = simulate_window(probs, closes, atr_vals, hold_bars, sl_mult)
        all_pnl.extend(pnl.tolist())

    all_pnl = np.array(all_pnl)
    wf_return = all_pnl.sum()
    wf_std    = all_pnl.std() if all_pnl.std() > 0 else 1e-9
    wf_sharpe = (all_pnl.mean() / wf_std) * ANNUAL_FACTOR
    wf_equity = np.cumsum(all_pnl)

    return dict(
        symbol=symbol,
        timeframe=tf_name,
        hold_bars=hold_bars,
        sl_mult=round(sl_mult, 2),
        wf_return=round(wf_return, 6),
        wf_sharpe_sl=round(wf_sharpe, 4),
        wf_std=round(wf_std, 8),
        wf_equity=wf_equity.tolist(),
        wf_pnl=all_pnl.tolist(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. ОТБОР СТРАТЕГИЙ И ФОРМИРОВАНИЕ АНСАМБЛЯ
# ─────────────────────────────────────────────────────────────────────────────
def select_ensemble(results: list[dict]) -> pd.DataFrame:
    """
    Отобрать стратегии по Sharpe, декоррелировать, рассчитать веса.
    """
    rdf = pd.DataFrame(results)
    rdf = rdf[(rdf["wf_sharpe_sl"] >= SHARPE_THRESHOLD) & (rdf["wf_return"] > 0)]
    rdf = rdf.sort_values("wf_sharpe_sl", ascending=False).reset_index(drop=True)

    if rdf.empty:
        print("[WARN] Нет стратегий, прошедших фильтр Sharpe!")
        return pd.DataFrame()

    # Декорреляция
    pnl_matrix = np.array(rdf["wf_pnl"].tolist())
    selected_idx = [0]

    for i in range(1, len(rdf)):
        correlated = False
        for j in selected_idx:
            corr = np.corrcoef(pnl_matrix[i], pnl_matrix[j])[0, 1]
            if abs(corr) > CORR_THRESHOLD:
                correlated = True
                break
        if not correlated:
            selected_idx.append(i)

    ensemble = rdf.iloc[selected_idx].copy()

    # Инверсно-волатильные веса
    inv_vol = 1.0 / ensemble["wf_std"]
    ensemble["weight"] = (inv_vol / inv_vol.sum()).round(4)

    print(f"[INFO] Отобрано {len(ensemble)} стратегий из {len(rdf)} "
          f"(после декорреляции из {len(results)} всего)")

    return ensemble


def plot_ensemble(ensemble: pd.DataFrame, save_path: str = "walk_forward_ensemble.png"):
    """Построить график совокупной кумулятивной доходности."""
    if ensemble.empty:
        return

    weighted_equity = None
    for _, row in ensemble.iterrows():
        eq = np.array(row["wf_equity"]) * row["weight"]
        if weighted_equity is None:
            weighted_equity = eq
        else:
            min_len = min(len(weighted_equity), len(eq))
            weighted_equity = weighted_equity[:min_len] + eq[:min_len]

    plt.figure(figsize=(14, 6))
    plt.plot(weighted_equity, linewidth=1.5, color="#00c853")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.title("Walk-Forward Ensemble V5 — Cumulative Equity", fontsize=14)
    plt.xlabel("Out-of-Sample Bars")
    plt.ylabel("Cumulative PnL (weighted)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] График сохранён: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_mt5()

    all_results = []
    combos = list(product(SYMBOLS, TIMEFRAMES.keys(), HOLD_RANGE, SL_MULT_RANGE))
    total = len(combos)
    print(f"[INFO] Всего комбинаций: {total}")

    for idx, (sym, tf, hb, slm) in enumerate(combos, 1):
        if idx % 50 == 0 or idx == 1:
            print(f"  [{idx}/{total}] {sym} {tf} hold={hb} sl={slm:.2f}")

        res = walk_forward(sym, tf, hb, slm)
        if res is not None:
            all_results.append(res)

    print(f"\n[INFO] Получено {len(all_results)} результатов")

    # Отбор
    ensemble = select_ensemble(all_results)

    if not ensemble.empty:
        # Сохранить CSV
        out_cols = ["symbol", "timeframe", "hold_bars", "sl_mult", "weight"]
        ensemble[out_cols].to_csv("walk_forward_strategies_v5.csv", index=False)
        print("[INFO] CSV сохранён: walk_forward_strategies_v5.csv")

        # График
        plot_ensemble(ensemble)

    mt5.shutdown()
    print("[INFO] Готово.")


if __name__ == "__main__":
    main()
