"""
diagnostic_optuna.py
Запуск: python diagnostic_optuna.py
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
SYMBOL    = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
N_BARS    = 15000
N_TRIALS  = 500
MIN_TRADES_HARD = 1

# ════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════
def load_mt5(symbol, tf, n_bars):
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Symbol {symbol} not found")
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise RuntimeError("No data")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    if "EUR" in symbol or "USD" in symbol:
        df = df[df.index.dayofweek < 5]
    return df[["open","high","low","close","tick_volume"]].copy()


def enrich_candles(df, atr_period=14):
    df = df.copy()
    df["range"]     = df["high"] - df["low"]
    df["body"]      = df["close"] - df["open"]
    df["abs_body"]  = df["body"].abs()
    df["direction"] = np.sign(df["body"]).astype(int)
    df["body_pct"]  = (df["abs_body"] / df["range"].replace(0, np.nan)).fillna(0)
    df["upper_wick"]= df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"]= df[["open","close"]].min(axis=1) - df["low"]
    df["upper_wick_pct"] = (df["upper_wick"] / df["range"].replace(0, np.nan)).fillna(0)
    df["lower_wick_pct"] = (df["lower_wick"] / df["range"].replace(0, np.nan)).fillna(0)
    df["wick_asymm"]= df["upper_wick_pct"] - df["lower_wick_pct"]

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"]       = tr.rolling(atr_period).mean()
    df["range_atr"] = df["range"] / df["atr"].replace(0, np.nan)
    df["body_atr"]  = df["abs_body"] / df["atr"].replace(0, np.nan)
    df["hour"]      = df.index.hour
    df["dow"]       = df.index.dayofweek
    return df.dropna()


# ════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════
def build_features(df, K=3):
    cols = ["direction","body_pct","range_atr","upper_wick_pct",
            "lower_wick_pct","wick_asymm","body_atr"]
    feat = pd.DataFrame(index=df.index)
    for lag in range(K):
        for c in cols:
            feat[f"{c}_lag{lag}"] = df[c].shift(lag)
    feat["body_sum"] = sum(df["body"].shift(i) for i in range(K))
    feat["dir_sum"]  = sum(df["direction"].shift(i) for i in range(K))
    feat["atr_avg"]  = feat[[f"range_atr_lag{i}" for i in range(K)]].mean(axis=1)
    feat["hour"]     = df["hour"]
    feat["dow"]      = df["dow"]
    return feat


def compute_target(df, horizon=6):
    fut_max = df["high"].shift(-1).rolling(horizon, min_periods=1).max()
    fut_min = df["low"].shift(-1).rolling(horizon, min_periods=1).min()
    long_p  = (fut_max - df["close"]) * 10000
    short_p = (df["close"] - fut_min) * 10000
    tgt = np.where(long_p >= short_p, long_p, -short_p)
    tgt = pd.Series(tgt, index=df.index)
    tgt.iloc[-horizon:] = np.nan
    return tgt


def run_core(df, params):
    K         = params["K"]
    horizon   = params["horizon"]
    threshold = params["threshold_pips"]
    min_body  = params["min_body_pct"]
    min_ratr  = params["min_range_atr"]
    hold      = params["hold_bars"]
    tp_p      = params["tp_pips"]
    sl_p      = params["sl_pips"]
    sess_s    = params["session_start"]
    sess_e    = params["session_end"]

    feat = build_features(df, K=K)
    tgt  = compute_target(df, horizon=horizon)

    data = feat.copy()
    data["tgt"] = tgt
    data = data.dropna()

    diag = {
        "n_data": len(data),
        "n_signals": 0,
        "n_trades": 0,
        "fail_reason": None,
        "pred_abs_mean": 0.0,
        "pred_abs_max": 0.0,
        "pred_abs_p95": 0.0,
    }

    if len(data) < 200:
        diag["fail_reason"] = f"too_few_data({len(data)})"
        return None, diag

    split = int(len(data) * 0.65)
    train, test = data.iloc[:split], data.iloc[split:]
    feat_cols = [c for c in data.columns if c != "tgt"]

    mdl = lgb.LGBMRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        min_child_samples=5, num_leaves=15, verbose=-1, n_jobs=-1,
    )
    try:
        mdl.fit(train[feat_cols], train["tgt"])
    except Exception as e:
        diag["fail_reason"] = f"model_fit_error({e})"
        return None, diag

    test = test.copy()
    test["pred"] = mdl.predict(test[feat_cols])

    diag["pred_abs_mean"] = float(test["pred"].abs().mean())
    diag["pred_abs_max"]  = float(test["pred"].abs().max())
    diag["pred_abs_p95"]  = float(test["pred"].abs().quantile(0.95))

    # ★ ФИКС: берём только нужные колонки из df, без hour/dow (они уже в feat)
    df_join = df[["open","high","low","close","body_pct","range_atr"]].copy()

    # join по индексу — у test индекс из data, у df_join индекс df
    test_full = test.join(df_join, how="left")

    mask = (
        (test_full["pred"].abs() >= threshold) &
        (test_full["body_pct"]   >= min_body)  &
        (test_full["range_atr"]  >= min_ratr)  &
        (test_full["hour"]       >= sess_s)     &  # hour уже в test через feat
        (test_full["hour"]       <= sess_e)
    )
    signals = test_full[mask].copy()
    diag["n_signals"] = len(signals)

    if len(signals) == 0:
        diag["fail_reason"] = (
            f"no_signals | "
            f"thr={threshold:.1f} pred_max={diag['pred_abs_max']:.2f} pred_p95={diag['pred_abs_p95']:.2f} | "
            f"min_body={min_body:.2f} body_mean={test_full['body_pct'].mean():.2f} | "
            f"min_ratr={min_ratr:.2f} ratr_mean={test_full['range_atr'].mean():.2f} | "
            f"sess={sess_s}-{sess_e}"
        )
        return None, diag

    signals["dir"] = np.sign(signals["pred"])
    trades = []
    idx = df.index

    for entry_time, sig in signals.iterrows():
        try:
            i_e = idx.get_loc(entry_time)
        except KeyError:
            continue
        if i_e + 1 >= len(df):
            continue

        entry     = sig["close"]
        direction = int(sig["dir"])
        tp = entry + tp_p / 10000 * direction
        sl = entry - sl_p / 10000 * direction
        exit_i = min(i_e + hold, len(df) - 1)

        pnl = 0; reason = "TO"
        ep  = df["close"].iloc[exit_i]
        et  = df.index[exit_i]

        for j in range(i_e + 1, exit_i + 1):
            bar = df.iloc[j]
            if direction > 0:
                if bar["low"]  <= sl: pnl=(sl-entry)*10000; reason="SL"; ep=sl; et=df.index[j]; break
                if bar["high"] >= tp: pnl=(tp-entry)*10000; reason="TP"; ep=tp; et=df.index[j]; break
            else:
                if bar["high"] >= sl: pnl=(entry-sl)*10000; reason="SL"; ep=sl; et=df.index[j]; break
                if bar["low"]  <= tp: pnl=(entry-tp)*10000; reason="TP"; ep=tp; et=df.index[j]; break

        if reason == "TO":
            pnl = (ep - entry) * 10000 * direction

        trades.append({"pnl": pnl, "reason": reason})

    diag["n_trades"] = len(trades)

    if len(trades) < MIN_TRADES_HARD:
        diag["fail_reason"] = f"too_few_trades({len(trades)})"
        return None, diag

    tdf = pd.Series([t["pnl"] for t in trades])
    n   = len(tdf)
    wr  = float((tdf > 0).mean())
    mp  = float(tdf.mean())
    sp  = float(tdf.std()) if n > 1 else 1.0
    sh  = mp / sp * np.sqrt(n) if sp > 0 else 0.0
    total = float(tdf.sum())
    gw  = float(tdf[tdf > 0].sum())
    gl  = float(abs(tdf[tdf <= 0].sum()))
    pf  = gw / gl if gl > 0 else 999.0

    return {
        "n_trades": n, "wr": wr, "sharpe": sh,
        "pf": pf, "total_pips": total, "params": params
    }, diag


# ════════════════════════════════════════════
# OPTUNA
# ════════════════════════════════════════════
_stats = {
    "calls": 0,
    "no_signals": 0,
    "too_few_trades": 0,
    "other_fail": 0,
    "success": 0,
    "no_signals_reasons": [],
}
_best_score  = [-999.0]
_best_diag   = [None]
_best_params = [None]
_best_result = [None]


def objective(trial):
    params = {
        "K":              trial.suggest_int("K", 1, 8),
        "horizon":        trial.suggest_int("horizon", 1, 20),
        "threshold_pips": trial.suggest_float("threshold_pips", 0.5, 50.0),
        "min_body_pct":   trial.suggest_float("min_body_pct", 0.0, 0.95),
        "min_range_atr":  trial.suggest_float("min_range_atr", 0.0, 3.0),
        "hold_bars":      trial.suggest_int("hold_bars", 1, 30),
        "tp_pips":        trial.suggest_float("tp_pips", 2.0, 60.0),
        "sl_pips":        trial.suggest_float("sl_pips", 2.0, 60.0),
        "session_start":  trial.suggest_int("session_start", 0, 11),
        "session_end":    trial.suggest_int("session_end", 12, 23),
    }
    if params["session_end"] <= params["session_start"]:
        params["session_end"] = params["session_start"] + 2

    _stats["calls"] += 1

    try:
        result, diag = run_core(df_train, params)
    except Exception as e:
        _stats["other_fail"] += 1
        print(f"  ⚠ Exception in trial: {e}")
        return -999.0

    if result is None:
        reason = diag.get("fail_reason", "unknown")
        if "no_signals" in reason:
            _stats["no_signals"] += 1
            if len(_stats["no_signals_reasons"]) < 30:
                _stats["no_signals_reasons"].append(reason)
        elif "too_few_trades" in reason:
            _stats["too_few_trades"] += 1
        else:
            _stats["other_fail"] += 1
        return -999.0

    _stats["success"] += 1

    score = (
        result["sharpe"] * 1.0
        + np.log(max(result["n_trades"], 1)) * 0.3
        + (0.5 if result["pf"] > 1.2 else 0.0)
        - (3.0 if result["total_pips"] < 0 else 0.0)
    )

    if score > _best_score[0]:
        _best_score[0]  = score
        _best_diag[0]   = diag
        _best_params[0] = params
        _best_result[0] = result
        print(
            f"\n  🏆 NEW BEST  score={score:.3f} | "
            f"n={result['n_trades']} wr={result['wr']*100:.0f}% "
            f"pf={result['pf']:.2f} pips={result['total_pips']:+.1f}\n"
            f"     K={params['K']} hor={params['horizon']} "
            f"thr={params['threshold_pips']:.2f} "
            f"body={params['min_body_pct']:.2f} ratr={params['min_range_atr']:.2f} "
            f"hold={params['hold_bars']} tp={params['tp_pips']:.1f} sl={params['sl_pips']:.1f} "
            f"sess={params['session_start']}-{params['session_end']}"
        )

    return score


def print_progress(study, trial):
    i = _stats["calls"]
    if i % 50 == 0:
        print(
            f"[{i:>4}/{N_TRIALS}] "
            f"✅ success={_stats['success']:>4} | "
            f"❌ no_signals={_stats['no_signals']:>4} | "
            f"❌ few_trades={_stats['too_few_trades']:>4} | "
            f"❌ other={_stats['other_fail']:>3} | "
            f"best={_best_score[0]:.3f}"
        )


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == "__main__":

    print(f"Loading {SYMBOL} ...")
    raw = load_mt5(SYMBOL, TIMEFRAME, N_BARS)
    df  = enrich_candles(raw)
    print(f"Ready: {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")

    oos_bars = min(30 * 24, len(df) // 3)
    df_train = df.iloc[:-oos_bars]
    df_oos   = df.iloc[-oos_bars:]
    print(f"Train: {len(df_train)} bars | OOS: {len(df_oos)} bars\n")

    # ── Sanity check ──────────────────────────────────────
    print("── Sanity check (threshold=1.0, все фильтры=0) ──")
    soft = {
        "K": 2, "horizon": 5,
        "threshold_pips": 1.0,
        "min_body_pct": 0.0, "min_range_atr": 0.0,
        "hold_bars": 5, "tp_pips": 10.0, "sl_pips": 10.0,
        "session_start": 0, "session_end": 23,
    }
    res_s, diag_s = run_core(df_train, soft)
    print(f"  n_data        = {diag_s['n_data']}")
    print(f"  pred_abs_mean = {diag_s['pred_abs_mean']:.4f}  ← если < 1, модель слабая")
    print(f"  pred_abs_max  = {diag_s['pred_abs_max']:.4f}")
    print(f"  pred_abs_p95  = {diag_s['pred_abs_p95']:.4f}")
    print(f"  n_signals     = {diag_s['n_signals']}")
    print(f"  n_trades      = {diag_s['n_trades']}")
    if diag_s["fail_reason"]:
        print(f"  fail_reason   = {diag_s['fail_reason']}")
    if res_s:
        print(f"  ✅ WORKS: n={res_s['n_trades']} wr={res_s['wr']*100:.0f}% "
              f"sh={res_s['sharpe']:.2f} pf={res_s['pf']:.2f} pips={res_s['total_pips']:+.1f}")
    else:
        print(f"  ❌ None даже с порогом=1.0")
        print(f"     Если pred_abs_max < 1.0 → LightGBM regularized to near-zero")
        print(f"     Попробуй другой символ или M15")

    # ── Optuna ────────────────────────────────────────────
    print(f"\n── Optuna {N_TRIALS} trials (n_jobs=1) ──")
    sampler = TPESampler(seed=42, n_startup_trials=100)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, callbacks=[print_progress])

    # ── Итоги ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("ИТОГИ")
    print("═" * 60)
    print(f"Trials total  : {_stats['calls']}")
    print(f"✅ success     : {_stats['success']}")
    print(f"❌ no_signals  : {_stats['no_signals']}")
    print(f"❌ few_trades  : {_stats['too_few_trades']}")
    print(f"❌ other       : {_stats['other_fail']}")

    if _stats["success"] == 0:
        print("\n⚠️  0 рабочих наборов параметров!")
        print("\nПримеры причин no_signals:")
        for i, r in enumerate(_stats["no_signals_reasons"][:10]):
            print(f"  [{i+1}] {r}")

        print("\n── Интерпретация ──")
        print("  pred_abs_max << threshold  → порог слишком высокий")
        print("  pred_abs_mean < 2          → LightGBM не видит сигнал в данных")
        print("  body_mean << min_body      → фильтр тела отрезает всё")
        print("  ratr_mean << min_ratr      → фильтр ATR отрезает всё")
        print("\n── Следующий шаг ──")
        print("  Измени SYMBOL = 'XAUUSD' или TIMEFRAME = mt5.TIMEFRAME_M15")
        print("  и запусти снова")

    else:
        print(f"\n🏆 Best score  : {_best_score[0]:.3f}")
        print(f"Best result   : n={_best_result[0]['n_trades']} "
              f"wr={_best_result[0]['wr']*100:.0f}% "
              f"pf={_best_result[0]['pf']:.2f} "
              f"pips={_best_result[0]['total_pips']:+.1f}")
        print(f"Best params   :")
        for k, v in _best_params[0].items():
            print(f"  {k:20s} = {v}")

        print("\n── Top-5 trials ──")
        top5 = sorted(
            [t for t in study.trials if t.value is not None and t.value > -900],
            key=lambda t: t.value, reverse=True
        )[:5]
        for rank, t in enumerate(top5, 1):
            print(f"  #{rank} score={t.value:.3f} | {t.params}")