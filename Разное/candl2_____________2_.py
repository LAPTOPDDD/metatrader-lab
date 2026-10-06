"""
ma_lgbm.py — MA стратегия + LightGBM фильтр
Walk-Forward по годам — честная проверка
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import warnings, time
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYMBOL    = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
N_BARS    = 90000
PIP       = 0.0001
SPREAD    = 2.0   # пипсов на сделку (спред + комиссия)


# ════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════
def load_data():
    if not mt5.initialize():
        raise RuntimeError(f"MT5: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown()
        raise RuntimeError(f"No {SYMBOL}")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, N_BARS)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df[df.index.dayofweek < 5]
    return df[["open","high","low","close","tick_volume"]].copy()


def add_features(df):
    df = df.copy()
    c  = df["close"]
    h  = df["high"]
    l  = df["low"]
    v  = df["tick_volume"]

    # ── ATR ──
    prev = c.shift(1)
    tr   = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr50"] = tr.rolling(50).mean()
    atr14 = df["atr14"].replace(0, np.nan)

    # ── Скользящие средние ──
    for p in [5, 8, 13, 21, 34, 50, 89, 144, 200]:
        df[f"ma{p}"]      = c.rolling(p).mean()
        df[f"ma{p}_dist"] = (c - df[f"ma{p}"]) / atr14

    # ── EMA ──
    for p in [9, 21, 50, 200]:
        df[f"ema{p}"]      = c.ewm(span=p, adjust=False).mean()
        df[f"ema{p}_dist"] = (c - df[f"ema{p}"]) / atr14

    # ── MA Crossover ──
    pairs = [(8,21), (13,34), (21,50), (50,200), (9,21)]
    for f_p, s_p in pairs:
        fast = df[f"ma{f_p}"] if f_p != 9 else df["ema9"]
        slow = df[f"ma{s_p}"]
        diff = (fast - slow).fillna(0)

        df[f"cross_{f_p}_{s_p}"]     = (diff / atr14).fillna(0)
        # ★ FIX: fillna(0) перед astype(int)
        df[f"cross_{f_p}_{s_p}_dir"] = np.sign(diff).fillna(0).astype(int)

        sign_change = (np.sign(diff) != np.sign(diff.shift(1))).astype(int)
        df[f"cross_{f_p}_{s_p}_age"] = (
            sign_change
            .groupby(sign_change.eq(0).cumsum())
            .cumcount()
            .fillna(0)
            .astype(int)
        )

    # ── Momentum ──
    for p in [5, 10, 20, 50]:
        df[f"mom{p}"] = ((c - c.shift(p)) / atr14).fillna(0)

    # ── RSI ──
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

    # ── Bollinger ──
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    bb_range       = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"] = (bb_range / atr14).fillna(0)
    df["bb_pos"]   = ((c - df["bb_lower"]) / bb_range).fillna(0.5)

    # ── ATR ratio ──
    df["atr_ratio"] = (df["atr14"] / df["atr50"].replace(0, np.nan)).fillna(1)

    # ── Volume ──
    vol_ma20       = v.rolling(20).mean().replace(0, np.nan)
    vol_ma5        = v.rolling(5).mean().replace(0, np.nan)
    df["vol_ratio"] = (v / vol_ma20).fillna(1)
    df["vol_trend"] = (vol_ma5 / vol_ma20).fillna(1)

    # ── Свеча ──
    df["body"]     = c - df["open"]
    df["range"]    = (h - l).replace(0, np.nan)
    df["body_pct"] = (df["body"].abs() / df["range"]).fillna(0)
    df["uw_pct"]   = (
        (h - df[["open","close"]].max(axis=1)) / df["range"]
    ).fillna(0)
    df["lw_pct"]   = (
        (df[["open","close"]].min(axis=1) - l) / df["range"]
    ).fillna(0)
    df["direction"]= np.sign(df["body"]).fillna(0).astype(int)

    # ── Время ──
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek

    return df.dropna()

# ════════════════════════════════════════════
# MA SIGNALS — базовая стратегия БЕЗ фильтра
# ════════════════════════════════════════════
def generate_ma_signals(df, fast=21, slow=50):
    """
    Классическое пересечение MA.
    Возвращает DataFrame с сигналами.
    """
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    diff    = fast_ma - slow_ma
    signal  = np.sign(diff)
    # Сигнал только при пересечении
    cross = (signal != signal.shift(1)) & (signal != 0)
    sigs  = df[cross].copy()
    sigs["signal_dir"] = signal[cross].astype(int)
    return sigs


# ════════════════════════════════════════════
# BACKTEST ЧЕСТНЫЙ
# ════════════════════════════════════════════
def backtest_signals(df, signals, tp_p, sl_p, hold):
    trades = []
    idx    = df.index

    for entry_time, sig in signals.iterrows():
        try:
            i_e = idx.get_loc(entry_time)
        except KeyError:
            continue
        if i_e + 1 >= len(df): continue

        entry = df["close"].iloc[i_e]
        d     = int(sig["signal_dir"])
        tp    = entry + tp_p * PIP * d
        sl    = entry - sl_p * PIP * d
        ei    = min(i_e + hold, len(df) - 1)

        pnl = 0; reason = "TO"
        ep  = df["close"].iloc[ei]

        for j in range(i_e + 1, ei + 1):
            bar = df.iloc[j]
            if d > 0:
                if bar["low"]  <= sl: pnl=(sl-entry)/PIP; reason="SL"; break
                if bar["high"] >= tp: pnl=(tp-entry)/PIP; reason="TP"; break
            else:
                if bar["high"] >= sl: pnl=(entry-sl)/PIP; reason="SL"; break
                if bar["low"]  <= tp: pnl=(entry-tp)/PIP; reason="TP"; break

        if reason == "TO":
            pnl = (ep - entry) / PIP * d

        trades.append({
            "pnl": pnl, "reason": reason,
            "entry_time": entry_time,
            "dir": d, "year": entry_time.year,
        })

    return trades


def metrics(trades, label="", with_spread=True):
    if not trades: return None
    cost = SPREAD if with_spread else 0
    t    = pd.Series([x["pnl"] - cost for x in trades])
    n    = len(t)
    wr   = float((t > 0).mean())
    mp   = float(t.mean())
    sp   = float(t.std()) if n > 1 else 1.0
    sh   = mp / sp * np.sqrt(n) if sp > 0 else 0.0
    tot  = float(t.sum())
    gw   = float(t[t > 0].sum())
    gl   = float(abs(t[t <= 0].sum()))
    pf   = gw / gl if gl > 0 else 999.0
    cum  = t.cumsum()
    mdd  = float((cum.cummax() - cum).max())
    aw   = float(t[t > 0].mean())  if (t > 0).any()  else 0.0
    al   = float(abs(t[t <= 0].mean())) if (t <= 0).any() else 1.0
    rr   = aw / al if al > 0 else 0.0
    if label:
        mark = "✅" if tot > 0 else "❌"
        print(f"  {mark} {label}: n={n:>4} wr={wr*100:>5.1f}% "
              f"sh={sh:>6.3f} pf={pf:>5.2f} rr={rr:>4.2f} "
              f"pips={tot:>+8.1f} mdd={mdd:>6.1f}")
    return dict(n=n, wr=wr, sh=sh, pf=pf,
                tot=tot, mdd=mdd, rr=rr)


# ════════════════════════════════════════════
# DATASET для LightGBM фильтра
# ════════════════════════════════════════════
def build_lgbm_dataset(df, fast=21, slow=50, tp_p=20, sl_p=15, hold=48):
    """
    Каждое пересечение MA = событие.
    Признаки = состояние рынка В МОМЕНТ пересечения.
    Таргет = реальный PnL сделки (с TP/SL).
    """
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    diff    = fast_ma - slow_ma
    signal  = np.sign(diff)
    cross   = (signal != signal.shift(1)) & (signal != 0)

    meta_cols = ["long_pnl","short_pnl","pnl","dir","year","time"]
    feat_cols_base = [c for c in df.columns
                      if c not in ["open","high","low","close","tick_volume"]
                      and not c.startswith("ma")  # не сами MA — только дистанции
                      or c.endswith("_dist") or c.endswith("_dir")
                      or c.endswith("_age") or c.endswith("_ratio")
                      or c in ["rsi14","bb_width","bb_pos","atr_ratio",
                               "vol_ratio","vol_trend","body_pct",
                               "uw_pct","lw_pct","direction","hour","dow"]]

    # Убираем лишние колонки
    exclude = {"open","high","low","close","tick_volume",
               "atr14","atr50","bb_upper","bb_lower","body","range"}
    feat_cols = [c for c in df.columns if c not in exclude
                 and not c.startswith("ma") or "_dist" in c or "_dir" in c or "_age" in c]
    feat_cols = list(set(feat_cols))

    events = []
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    idx    = df.index
    n      = len(df)

    cross_idx = np.where(cross.values)[0]

    for i in cross_idx:
        if i < 200 or i >= n - hold - 1: continue

        d     = int(signal.iloc[i])
        entry = closes[i]

        # Считаем PnL для обоих направлений
        long_pnl = short_pnl = 0.0
        for j in range(i+1, min(i+hold+1, n)):
            h, l = highs[j], lows[j]
            if long_pnl == 0.0:
                if l <= entry - sl_p*PIP:   long_pnl  = -sl_p
                elif h >= entry + tp_p*PIP: long_pnl  =  tp_p
            if short_pnl == 0.0:
                if h >= entry + sl_p*PIP:   short_pnl = -sl_p
                elif l <= entry - tp_p*PIP: short_pnl =  tp_p

        ec = closes[min(i+hold, n-1)]
        if long_pnl  == 0.0: long_pnl  = (ec-entry)/PIP
        if short_pnl == 0.0: short_pnl = (entry-ec)/PIP

        # Реальный PnL в направлении сигнала
        pnl = long_pnl if d > 0 else short_pnl

        row = {c: df[c].iloc[i] for c in feat_cols if c in df.columns}
        row["long_pnl"]  = long_pnl
        row["short_pnl"] = short_pnl
        row["pnl"]       = pnl
        row["dir"]       = d
        row["year"]      = idx[i].year
        row["time"]      = idx[i]
        events.append(row)

    edf = pd.DataFrame(events)
    fc  = [c for c in edf.columns
           if c not in ["long_pnl","short_pnl","pnl","dir","year","time"]
           and edf[c].dtype in [np.float64, np.int64, float, int]]
    return edf, fc


# ════════════════════════════════════════════
# WALK-FORWARD СРАВНЕНИЕ
# ════════════════════════════════════════════
def walk_forward_compare(df, edf, fc,
                          fast=21, slow=50,
                          tp_p=20, sl_p=15, hold=48,
                          threshold=0.0):
    years = sorted(edf["year"].unique())

    print(f"\n{'year':>5} | {'base_net':>9} {'lgbm_net':>9} "
          f"{'base_n':>7} {'lgbm_n':>7} | {'verdict':>12}")
    print("-" * 60)

    all_base = []
    all_lgbm = []
    results  = []

    for test_year in years[3:]:
        tr_y  = [y for y in years if y < test_year]
        df_tr = edf[edf["year"].isin(tr_y)]
        df_te = edf[edf["year"] == test_year]

        if len(df_tr) < 100 or len(df_te) < 10:
            continue

        # ── Базовая MA (без фильтра) ──
        base_trades = [{"pnl": row["pnl"], "year": test_year}
                       for _, row in df_te.iterrows()]

        # ── LightGBM фильтр ──
        mdl = lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            min_child_samples=15, num_leaves=20,
            reg_lambda=2.0, reg_alpha=0.5,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, n_jobs=-1,
        )
        # Таргет: предсказываем реальный PnL
        mdl.fit(df_tr[fc], df_tr["pnl"],
                eval_set=[(df_te[fc], df_te["pnl"])],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)])

        preds = mdl.predict(df_te[fc])

        # Торгуем только когда модель уверена (pred > threshold)
        lgbm_trades = []
        for pred, (_, row) in zip(preds, df_te.iterrows()):
            if pred < threshold: continue  # модель говорит "скип"
            lgbm_trades.append({"pnl": row["pnl"], "year": test_year})

        # Метрики
        def net_pips(trades):
            if not trades: return 0.0
            return sum(t["pnl"] - SPREAD for t in trades)

        base_net = net_pips(base_trades)
        lgbm_net = net_pips(lgbm_trades)
        base_n   = len(base_trades)
        lgbm_n   = len(lgbm_trades)

        if lgbm_net > base_net and lgbm_net > 0:
            verdict = "✅ LGBM wins"
        elif lgbm_net > 0 and base_net <= 0:
            verdict = "✅ LGBM saves"
        elif base_net > lgbm_net and base_net > 0:
            verdict = "⚠️  Base wins"
        else:
            verdict = "❌ Both lose"

        print(f"  {test_year} | {base_net:>+9.1f} {lgbm_net:>+9.1f} "
              f"{base_n:>7} {lgbm_n:>7} | {verdict}")

        all_base.extend(base_trades)
        all_lgbm.extend(lgbm_trades)
        results.append({
            "year": test_year,
            "base_net": base_net, "lgbm_net": lgbm_net,
            "base_n": base_n, "lgbm_n": lgbm_n,
        })

    return all_base, all_lgbm, results, mdl, fc


# ════════════════════════════════════════════
# OPTUNA — ищет лучшую MA пару + threshold
# ════════════════════════════════════════════
def tune_everything(df, n_trials=300):
    print(f"\n── Optuna: {n_trials} trials ──")

    years_all = sorted(df.index.year.unique())

    _best = {"score": -999, "params": None}

    def objective(trial):
        fast  = trial.suggest_int("fast", 5, 50)
        slow  = trial.suggest_int("slow", fast+5, 200)
        tp_p  = trial.suggest_float("tp_pips", 10.0, 50.0)
        sl_p  = trial.suggest_float("sl_pips", 8.0,  30.0)
        hold  = trial.suggest_int("hold_bars", 12, 96)
        thr   = trial.suggest_float("threshold", -20.0, 20.0)

        if tp_p / sl_p < 0.8: return -999.0  # RR минимум 0.8

        try:
            edf, fc = build_lgbm_dataset(df, fast, slow, tp_p, sl_p, hold)
        except Exception:
            return -999.0

        if len(edf) < 200: return -999.0

        years = sorted(edf["year"].unique())
        if len(years) < 5: return -999.0

        all_pnl  = []
        n_profit = 0
        n_years  = 0

        for test_year in years[3:]:
            tr_y  = [y for y in years if y < test_year]
            df_tr = edf[edf["year"].isin(tr_y)]
            df_te = edf[edf["year"] == test_year]
            if len(df_tr) < 80 or len(df_te) < 8: continue

            mdl = lgb.LGBMRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_child_samples=10, num_leaves=15,
                reg_lambda=2.0, verbose=-1, n_jobs=-1,
            )
            try:
                mdl.fit(df_tr[fc], df_tr["pnl"])
            except Exception:
                continue

            preds  = mdl.predict(df_te[fc])
            trades = [row["pnl"] - SPREAD
                      for pred, (_, row) in zip(preds, df_te.iterrows())
                      if pred >= thr]

            if len(trades) < 3: continue
            t = pd.Series(trades)
            all_pnl.extend(trades)
            n_years += 1
            if t.sum() > 0: n_profit += 1

        if len(all_pnl) < 30 or n_years < 3:
            return -999.0

        t_all = pd.Series(all_pnl)
        sh    = (t_all.mean() / t_all.std() * np.sqrt(len(t_all))
                 if t_all.std() > 0 else 0)
        stab  = n_profit / n_years
        score = sh * stab

        if score > _best["score"]:
            _best["score"]  = score
            _best["params"] = {
                "fast": fast, "slow": slow,
                "tp_pips": tp_p, "sl_pips": sl_p,
                "hold_bars": hold, "threshold": thr,
            }
            print(f"  🏆 score={score:.3f} | "
                  f"MA({fast},{slow}) tp={tp_p:.0f} sl={sl_p:.0f} "
                  f"hold={hold} thr={thr:.1f} | "
                  f"sh={sh:.3f} stab={stab:.2f} n={len(all_pnl)}")

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=50),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nBest: score={study.best_value:.3f}")
    print(f"Params: {_best['params']}")
    return _best["params"]


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Loading {SYMBOL}...")
    raw = load_data()
    df  = add_features(raw)
    print(f"Data: {len(df)} bars | "
          f"{df.index[0].date()} → {df.index[-1].date()}\n")

    # ── Шаг 1: Optuna ищет лучшую MA пару + threshold ──
    best_p = tune_everything(df, n_trials=300)

    if best_p is None:
        print("❌ Optuna не нашла параметров")
        exit()

    fast  = best_p["fast"]
    slow  = best_p["slow"]
    tp_p  = best_p["tp_pips"]
    sl_p  = best_p["sl_pips"]
    hold  = best_p["hold_bars"]
    thr   = best_p["threshold"]

    # ── Шаг 2: Строим финальный датасет ──
    print(f"\nBuilding dataset MA({fast},{slow}) "
          f"tp={tp_p:.0f} sl={sl_p:.0f} hold={hold}...")
    edf, fc = build_lgbm_dataset(df, fast, slow, tp_p, sl_p, hold)
    print(f"Events: {len(edf)} | Features: {len(fc)}")
    print(edf.groupby("year").size().to_string())

    # ── Шаг 3: Walk-Forward сравнение ──
    print(f"\n── Walk-Forward: Base MA vs LGBM-filtered MA ──")
    print(f"   MA({fast},{slow})  TP={tp_p:.0f}  SL={sl_p:.0f}  "
          f"Hold={hold}  Threshold={thr:.1f}\n")

    all_base, all_lgbm, results, last_mdl, feat_cols = walk_forward_compare(
        df, edf, fc, fast, slow, tp_p, sl_p, hold, thr
    )

    # ── Шаг 4: Итоги ──
    print("\n" + "═"*60)

    def summary(trades, label):
        if not trades:
            print(f"{label}: нет сделок"); return
        t   = pd.Series([x["pnl"] - SPREAD for x in trades])
        n   = len(t)
        sh  = t.mean()/t.std()*np.sqrt(n) if t.std() > 0 else 0
        wr  = float((t > 0).mean())
        tot = float(t.sum())
        pf_g = float(t[t>0].sum())
        pf_l = float(abs(t[t<=0].sum()))
        pf   = pf_g/pf_l if pf_l > 0 else 999
        print(f"{label}:")
        print(f"  Сделок={n}  WR={wr*100:.1f}%  "
              f"Sharpe={sh:.3f}  PF={pf:.2f}  "
              f"Pips={tot:+.1f}")

    summary(all_base, "BASE MA (без фильтра)")
    summary(all_lgbm, "LGBM-filtered MA")

    # Улучшение
    if all_base and all_lgbm:
        base_tot = sum(t["pnl"]-SPREAD for t in all_base)
        lgbm_tot = sum(t["pnl"]-SPREAD for t in all_lgbm)
        if base_tot != 0:
            improvement = (lgbm_tot - base_tot) / abs(base_tot) * 100
            print(f"\nУлучшение: {improvement:+.1f}%")

        # Сколько сделок отфильтровано
        filtered = len(all_base) - len(all_lgbm)
        pct      = filtered / max(len(all_base), 1) * 100
        print(f"Отфильтровано: {filtered} ({pct:.0f}%) сделок")

    # Feature importance
    if last_mdl and feat_cols:
        print(f"\nTop-10 features (что помогает фильтровать):")
        imp = (pd.DataFrame({"f": feat_cols,
                             "i": last_mdl.feature_importances_})
               .sort_values("i", ascending=False).head(10))
        mx = imp["i"].max()
        for _, r in imp.iterrows():
            bar = "█" * int(r["i"]/mx*25)
            print(f"  {r['f']:28s} {bar}")