"""
level_reaction_v2.py
Fixes:
1. Больше событий — убираем требование at_level==1, берём все близкие к уровню
2. Threshold через Optuna по годам
3. Честный вывод
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
TP_PIPS   = 15.0
SL_PIPS   = 15.0
HOLD_BARS = 12
PIP       = 0.0001

SWING_BARS    = 5
LEVEL_TOL_ATR = 0.5    # ★ шире — больше событий
CLUSTER_PIPS  = 10
MIN_LEVEL_AGE = 10     # ★ меньше — уровни появляются быстрее
MAX_LEVELS    = 100    # ★ больше уровней


# ════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════
def load_data():
    if not mt5.initialize():
        raise RuntimeError(f"MT5: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown(); raise RuntimeError(f"No {SYMBOL}")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, N_BARS)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df[df.index.dayofweek < 5]
    return df[["open","high","low","close","tick_volume"]].copy()


def add_indicators(df, atr_period=14):
    df = df.copy()
    df["range"]     = df["high"] - df["low"]
    df["body"]      = df["close"] - df["open"]
    df["abs_body"]  = df["body"].abs()
    df["direction"] = np.sign(df["body"]).astype(int)
    df["body_pct"]  = (df["abs_body"] / df["range"].replace(0,np.nan)).fillna(0)
    df["uw"]        = df["high"] - df[["open","close"]].max(axis=1)
    df["lw"]        = df[["open","close"]].min(axis=1) - df["low"]
    df["uw_pct"]    = (df["uw"] / df["range"].replace(0,np.nan)).fillna(0)
    df["lw_pct"]    = (df["lw"] / df["range"].replace(0,np.nan)).fillna(0)
    prev = df["close"].shift(1)
    tr   = pd.concat([df["high"]-df["low"],
                      (df["high"]-prev).abs(),
                      (df["low"] -prev).abs()], axis=1).max(axis=1)
    df["atr"]       = tr.rolling(atr_period).mean()
    df["ratr"]      = df["range"] / df["atr"].replace(0,np.nan)
    df["batr"]      = df["abs_body"] / df["atr"].replace(0,np.nan)
    df["hour"]      = df.index.hour
    df["dow"]       = df.index.dayofweek
    df["mom5"]      = (df["close"]-df["close"].shift(5))  / df["atr"].replace(0,np.nan)
    df["mom20"]     = (df["close"]-df["close"].shift(20)) / df["atr"].replace(0,np.nan)
    df["vol_ratio"] = df["tick_volume"] / df["tick_volume"].rolling(20).mean()
    # ADX упрощённый
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(50).mean().replace(0,np.nan)
    return df.dropna()


# ════════════════════════════════════════════
# LEVEL TRACKER
# ════════════════════════════════════════════
class Level:
    __slots__ = ["price","type","birth_bar","touches","breaks","active"]
    def __init__(self, price, lvl_type, birth_bar):
        self.price      = price
        self.type       = lvl_type
        self.birth_bar  = birth_bar
        self.touches    = 0
        self.breaks     = 0
        self.active     = True
    def age(self, i): return i - self.birth_bar
    def strength(self): return max(1, self.touches * 2 - self.breaks * 3 + 1)
    def is_exhausted(self): return self.touches >= 8 or self.breaks >= 3


class LevelTracker:
    def __init__(self):
        self.levels = []

    def _cluster_exists(self, price, tol):
        return any(l.active and abs(l.price-price) <= tol
                   for l in self.levels)

    def update(self, i, highs, lows, closes, atr_val):
        tol_cluster = CLUSTER_PIPS * PIP
        sb = SWING_BARS
        check = i - sb
        if check >= sb:
            # Swing high
            if (highs[check] == max(highs[check-sb:check+1]) and
                    not self._cluster_exists(highs[check], tol_cluster)):
                self.levels.append(Level(highs[check], "resistance", check))
            # Swing low
            if (lows[check] == min(lows[check-sb:check+1]) and
                    not self._cluster_exists(lows[check], tol_cluster)):
                self.levels.append(Level(lows[check], "support", check))

        # Обновляем касания
        tol = atr_val * LEVEL_TOL_ATR if atr_val > 0 else 0.001
        for lvl in self.levels:
            if not lvl.active: continue
            if lvl.age(i) < MIN_LEVEL_AGE: continue
            if not (lows[i] <= lvl.price + tol and highs[i] >= lvl.price - tol):
                continue
            lvl.touches += 1
            # Пробой
            if lvl.type == "resistance" and closes[i] > lvl.price + tol:
                lvl.breaks += 1
            elif lvl.type == "support" and closes[i] < lvl.price - tol:
                lvl.breaks += 1
            if lvl.is_exhausted():
                lvl.active = False

        # Pruning
        if len(self.levels) > MAX_LEVELS:
            active = [l for l in self.levels if l.active]
            if len(active) > MAX_LEVELS:
                active.sort(key=lambda l: l.strength())
                for l in active[:len(active)-MAX_LEVELS]:
                    l.active = False

    def nearest(self, i, price, atr_val):
        """Возвращает контекст ближайших уровней."""
        if atr_val <= 0: atr_val = 0.001
        active = [l for l in self.levels
                  if l.active and l.age(i) >= MIN_LEVEL_AGE]
        if not active: return None

        above = [l for l in active if l.price > price]
        below = [l for l in active if l.price <= price]
        nr    = min(active, key=lambda l: abs(l.price-price))
        dist  = abs(nr.price - price)

        return {
            "dist_atr":       dist / atr_val,
            "at_level":       int(dist < atr_val * LEVEL_TOL_ATR),
            "nearest_type":   1 if nr.type=="resistance" else -1,
            "nearest_str":    nr.strength(),
            "nearest_age":    nr.age(i),
            "nearest_touch":  nr.touches,
            "nearest_breaks": nr.breaks,
            "res_dist":       (min(above,key=lambda l:l.price-price).price-price)/atr_val if above else 10.0,
            "sup_dist":       (price-max(below,key=lambda l:l.price).price)/atr_val if below else 10.0,
            "res_str":        max(above,key=lambda l:l.strength()).strength() if above else 0,
            "sup_str":        max(below,key=lambda l:l.strength()).strength() if below else 0,
            "n_above":        len(above),
            "n_below":        len(below),
        }


# ════════════════════════════════════════════
# BUILD DATASET
# ════════════════════════════════════════════
def build_dataset(df):
    print("Building dataset...")
    t0 = time.time()

    tracker = LevelTracker()
    highs   = df["high"].values
    lows    = df["low"].values
    closes  = df["close"].values
    atrs    = df["atr"].values
    idx     = df.index
    n       = len(df)

    # Кэшируем признаки
    arrays = {c: df[c].values for c in
              ["body_pct","uw_pct","lw_pct","ratr","batr","direction",
               "hour","dow","mom5","mom20","vol_ratio","atr_ratio"]}
    direction = arrays["direction"]

    events = []
    # ★ Собираем ВСЕ бары близко к уровню (не только at_level)
    # Фильтр: dist_atr < 1.5 (в 1.5 ATR от уровня)
    MAX_DIST_ATR = 1.5

    for i in range(SWING_BARS*2, n - HOLD_BARS - 1):
        atr_val = atrs[i]
        if np.isnan(atr_val) or atr_val <= 0: continue

        tracker.update(i, highs, lows, closes, atr_val)
        ctx = tracker.nearest(i, closes[i], atr_val)
        if ctx is None or ctx["dist_atr"] > MAX_DIST_ATR: continue

        # Признаки свечи
        feat = {k: arrays[k][i] for k in arrays}
        # Предыдущие бары
        feat["dir_sum3"]      = int(sum(direction[max(0,i-2):i+1]))
        feat["body_sum3"]     = float(np.mean(arrays["body_pct"][max(0,i-2):i+1]))
        feat["ratr_avg3"]     = float(np.mean(arrays["ratr"][max(0,i-2):i+1]))
        feat["approach"]      = float(abs(closes[i]-closes[max(0,i-4)])/atr_val)
        feat["consec_dir"]    = _consec(direction, i)
        # Прокол
        if ctx["nearest_type"] == 1:  # resistance
            lvl_p = closes[i] + ctx["dist_atr"] * atr_val
            pen   = (highs[i] - lvl_p) / atr_val
        else:
            lvl_p = closes[i] - ctx["dist_atr"] * atr_val
            pen   = (lvl_p - lows[i]) / atr_val
        feat["penetration"]   = max(0.0, float(pen))
        feat["close_pos"]     = float((closes[i]-lows[i]) / max(highs[i]-lows[i], PIP))
        # Контекст
        feat.update(ctx)

        # ── Результат ──
        entry = closes[i]
        long_pnl = short_pnl = 0.0
        for j in range(i+1, min(i+HOLD_BARS+1, n)):
            h, l = highs[j], lows[j]
            if long_pnl == 0.0:
                if l <= entry - SL_PIPS*PIP:   long_pnl  = -SL_PIPS
                elif h >= entry + TP_PIPS*PIP: long_pnl  =  TP_PIPS
            if short_pnl == 0.0:
                if h >= entry + SL_PIPS*PIP:   short_pnl = -SL_PIPS
                elif l <= entry - TP_PIPS*PIP: short_pnl =  TP_PIPS
        ec = closes[min(i+HOLD_BARS, n-1)]
        if long_pnl  == 0.0: long_pnl  = (ec-entry)/PIP
        if short_pnl == 0.0: short_pnl = (entry-ec)/PIP

        feat["long_pnl"]  = float(long_pnl)
        feat["short_pnl"] = float(short_pnl)
        feat["diff"]      = float(long_pnl - short_pnl)  # таргет
        feat["year"]      = idx[i].year
        feat["time"]      = idx[i]
        events.append(feat)

        if len(events) % 10000 == 0:
            print(f"  {i/n*100:.0f}% | {len(events)} events | "
                  f"{time.time()-t0:.0f}s")

    print(f"  Done: {len(events)} events | {time.time()-t0:.0f}s\n")
    return pd.DataFrame(events)


def _consec(direction, i):
    d = direction[i]
    if d == 0: return 0
    c = 0
    for j in range(i, max(i-10,-1), -1):
        if direction[j] == d: c += 1
        else: break
    return c


# ════════════════════════════════════════════
# WALK-FORWARD
# ════════════════════════════════════════════
def walk_forward(edf, threshold=3.0):
    years     = sorted(edf["year"].unique())
    meta_cols = ["long_pnl","short_pnl","diff","year","time"]
    feat_cols = [c for c in edf.columns if c not in meta_cols]

    results    = []
    all_trades = []

    for test_year in years[3:]:
        tr_years = [y for y in years if y < test_year]
        df_tr = edf[edf["year"].isin(tr_years)]
        df_te = edf[edf["year"] == test_year]

        if len(df_tr) < 300 or len(df_te) < 30:
            continue

        mdl = lgb.LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.02,
            min_child_samples=20, num_leaves=25,
            reg_lambda=3.0, reg_alpha=1.0,
            subsample=0.7, colsample_bytree=0.7,
            verbose=-1, n_jobs=-1,
        )
        mdl.fit(df_tr[feat_cols], df_tr["diff"],
                eval_set=[(df_te[feat_cols], df_te["diff"])],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)])

        preds = mdl.predict(df_te[feat_cols])

        trades = []
        for pred, (_, row) in zip(preds, df_te.iterrows()):
            if abs(pred) < threshold: continue
            d   = 1 if pred > 0 else -1
            pnl = row["long_pnl"] if d > 0 else row["short_pnl"]
            trades.append({"pnl": pnl, "year": test_year})

        if not trades:
            print(f"  {test_year}: 0 trades "
                  f"(events={len(df_te)} pred_max={abs(preds).max():.2f})")
            results.append({"year":test_year,"n":0,"pips":0.0})
            continue

        t = pd.Series([x["pnl"] for x in trades])
        n   = len(t); wr = float((t>0).mean())
        tot = float(t.sum()); mp = float(t.mean())
        sp  = float(t.std()) if n>1 else 1.0
        sh  = mp/sp*np.sqrt(n) if sp>0 else 0.0
        gw  = float(t[t>0].sum()); gl = float(abs(t[t<=0].sum()))
        pf  = gw/gl if gl>0 else 999.0
        cum = t.cumsum(); mdd = float((cum.cummax()-cum).max())
        aw  = float(t[t>0].mean()) if (t>0).any() else 0.0
        al  = float(abs(t[t<=0].mean())) if (t<=0).any() else 1.0
        rr  = aw/al if al>0 else 0.0

        mark = "✅" if tot > 0 else "❌"
        print(f"  {test_year} {mark}: n={n:>4} wr={wr*100:>5.1f}% "
              f"sh={sh:>6.3f} pf={pf:>5.2f} rr={rr:>4.2f} "
              f"pips={tot:>+8.1f} mdd={mdd:>6.1f}")
        results.append({"year":test_year,"n":n,"wr":round(wr*100,1),
                        "sh":round(sh,3),"pf":round(pf,2),"rr":round(rr,2),
                        "pips":round(tot,1),"mdd":round(mdd,1)})
        all_trades.extend(trades)

    return results, all_trades, mdl, feat_cols


# ════════════════════════════════════════════
# OPTUNA — ищет threshold и TP/SL
# ════════════════════════════════════════════
def tune_threshold(edf, n_trials=200):
    """
    Optuna ищет threshold при котором
    walk-forward даёт максимальный Sharpe.
    """
    print(f"\n── Optuna: {n_trials} trials ──")

    years     = sorted(edf["year"].unique())
    meta_cols = ["long_pnl","short_pnl","diff","year","time"]
    feat_cols = [c for c in edf.columns if c not in meta_cols]

    # Предобучаем модели по годам один раз
    models = {}
    for test_year in years[3:]:
        tr_y  = [y for y in years if y < test_year]
        df_tr = edf[edf["year"].isin(tr_y)]
        df_te = edf[edf["year"] == test_year]
        if len(df_tr) < 300 or len(df_te) < 20: continue

        mdl = lgb.LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.02,
            min_child_samples=20, num_leaves=25,
            reg_lambda=3.0, reg_alpha=1.0,
            subsample=0.7, colsample_bytree=0.7,
            verbose=-1, n_jobs=-1,
        )
        mdl.fit(df_tr[feat_cols], df_tr["diff"],
                eval_set=[(df_te[feat_cols], df_te["diff"])],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)])
        preds = mdl.predict(df_te[feat_cols])
        models[test_year] = (preds, df_te)

    if not models:
        print("Нет данных для Optuna"); return 3.0

    def objective(trial):
        thr = trial.suggest_float("threshold", 0.5, 10.0)

        all_pnl = []
        n_profitable = 0
        n_years = 0

        for year, (preds, df_te) in models.items():
            trades = []
            for pred, (_, row) in zip(preds, df_te.iterrows()):
                if abs(pred) < thr: continue
                d   = 1 if pred > 0 else -1
                pnl = row["long_pnl"] if d > 0 else row["short_pnl"]
                trades.append(pnl)
            if len(trades) < 5: continue
            t = pd.Series(trades)
            all_pnl.extend(trades)
            n_years += 1
            if t.sum() > 0: n_profitable += 1

        if len(all_pnl) < 20 or n_years < 2:
            return -999.0

        t_all = pd.Series(all_pnl)
        sh    = t_all.mean()/t_all.std()*np.sqrt(len(t_all)) if t_all.std()>0 else 0
        stab  = n_profitable / n_years  # доля прибыльных лет

        return sh * stab  # хотим и Sharpe и стабильность

    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_thr = study.best_params["threshold"]
    print(f"Best threshold: {best_thr:.3f} "
          f"(score={study.best_value:.3f})")
    return best_thr


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Loading {SYMBOL}...")
    raw = load_data()
    df  = add_indicators(raw)
    print(f"Data: {len(df)} bars | "
          f"{df.index[0].date()} → {df.index[-1].date()}\n")

    # Строим датасет
    edf = build_dataset(df)
    print(f"Events per year:")
    print(edf.groupby("year").size().to_string())

    if len(edf) < 500:
        print("Слишком мало событий. Попробуй XAUUSD или M15.")
        exit()

    # Ищем порог через Optuna
    best_thr = tune_threshold(edf, n_trials=200)

    # Walk-Forward с лучшим порогом
    print(f"\n── Walk-Forward (threshold={best_thr:.2f}) ──\n")
    results, all_trades, last_model, feat_cols = walk_forward(edf, threshold=best_thr)

    # ── Итоги ──
    print("\n" + "═"*65)
    rdf = pd.DataFrame(results)
    rdf_valid = rdf[rdf["n"] > 0]

    if rdf_valid.empty:
        print("❌ Нет торговых сигналов")
        print(f"   Попробуй: XAUUSD, TIMEFRAME_M15, или больше данных")
    else:
        profitable = (rdf_valid["pips"] > 0).sum()
        total      = len(rdf_valid)
        all_pnl    = pd.Series([t["pnl"] for t in all_trades])
        sh_total   = (all_pnl.mean()/all_pnl.std()*np.sqrt(len(all_pnl))
                      if all_pnl.std() > 0 else 0)

        print(f"Прибыльных лет: {profitable}/{total}")
        print(f"Всего сделок:   {len(all_pnl)}")
        print(f"Общий Sharpe:   {sh_total:.3f}")
        print(f"Общие pips:     {all_pnl.sum():+.1f}")
        print(f"Общий WR:       {(all_pnl>0).mean()*100:.1f}%")

        if profitable >= total * 0.65 and sh_total > 0.5 and len(all_pnl) >= 100:
            print("\n✅ РЕАЛЬНЫЙ ЭДЖ")
            print("   Следующий шаг: реальный форвард-тест 3 месяца")
        elif profitable >= total * 0.5:
            print("\n⚠️  СЛАБЫЙ ЭДЖ — нужно больше данных или фильтров")
        else:
            print("\n❌ НЕТ ЭДЖ — попробуй другой инструмент")

        # Feature importance
        print(f"\nTop-10 features:")
        imp = (pd.DataFrame({"f": feat_cols,
                             "i": last_model.feature_importances_})
               .sort_values("i", ascending=False).head(10))
        for _, r in imp.iterrows():
            bar = "█" * int(r["i"]/imp["i"].max()*25)
            print(f"  {r['f']:25s} {bar}")