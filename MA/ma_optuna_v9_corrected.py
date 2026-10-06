"""
MA V9 - Meta-Optimization (CORRECTED ENTRY)
Fix: o[i] -> o[i+1] to eliminate lookahead bias
EURUSD | find optimal: N_champions x Train_period
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import optuna
from optuna.samplers import TPESampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, time
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYMBOL    = "EURUSD"
PIP       = 0.0001
TF_CONFIG = {
    "M5":  (mt5.TIMEFRAME_M5,  50000, 40),
    "M15": (mt5.TIMEFRAME_M15, 30000, 50),
    "M30": (mt5.TIMEFRAME_M30, 20000, 60),
    "H1":  (mt5.TIMEFRAME_H1,  20000, 80),
}
N_TRIALS     = 80
TEST_DAYS    = 30
STEP_DAYS    = 30
COST_PER_TRADE = 0.7

CHAMPION_COUNTS = [1, 3, 5, 7, 10, 15]
TRAIN_DAYS_LIST = [90, 180, 270, 365, 545]

def load_data(tf, n_bars):
    try: mt5.shutdown()
    except: pass
    if not mt5.initialize():
        raise RuntimeError(f"MT5: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown(); raise RuntimeError(f"No {SYMBOL}")
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n_bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df[df.index.dayofweek < 5]
    return df[["open","high","low","close"]].copy()

def precompute(df):
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    prev_c = np.roll(c,1); prev_c[0]=c[0]
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
    atr = pd.Series(tr).rolling(14).mean().values
    return o, h, l, c, atr

def run_fast(o, h, l, c, atr, fast, slow, entry, sl_atr, tp_atr):
    n = len(c)
    maf = pd.Series(c).rolling(fast).mean().values
    mas = pd.Series(c).rolling(slow).mean().values
    denom = np.where(atr>1e-10, atr, 1e-10)
    diff = (maf - mas) / denom
    trades = []
    pos=0; ep=sl=tp=0.0; ei=0
    s = max(fast, slow, 14)
    for i in range(s, n):
        d = diff[i]
        if pos==0:
            if np.isnan(d): continue
            if d > entry and (i+1) < n:
                pos=1; ep=o[i+1]; sl=ep-atr[i]*sl_atr; tp=ep+atr[i]*tp_atr; ei=i+1
            elif d < -entry and (i+1) < n:
                pos=-1; ep=o[i+1]; sl=ep+atr[i]*sl_atr; tp=ep-atr[i]*tp_atr; ei=i+1
        else:
            b = i-ei
            if pos==1:
                if   l[i]<=sl: trades.append(-(sl-ep)/PIP - COST_PER_TRADE); pos=0
                elif h[i]>=tp: trades.append((tp-ep)/PIP - COST_PER_TRADE);  pos=0
                elif d>-entry*0.3 or (b>=3 and d<=0): trades.append((c[i]-ep)/PIP - COST_PER_TRADE); pos=0
            else:
                if   h[i]>=sl: trades.append((ep-sl)/PIP - COST_PER_TRADE);  pos=0
                elif l[i]<=tp: trades.append((ep-tp)/PIP - COST_PER_TRADE);  pos=0
                elif d>entry*0.3 or (b>=3 and d>=0): trades.append((ep-c[i])/PIP - COST_PER_TRADE); pos=0
    return np.array(trades) if trades else np.array([])

def objective(trial, o, h, l, c, atr, max_slow):
    fast = trial.suggest_int("fast",3,max(3,max_slow-11))
    slow = trial.suggest_int("slow",fast+10,max_slow)
    entry = trial.suggest_float("entry",0.05,1.2)
    sl = trial.suggest_float("sl_atr",0.5,4.0)
    tp = trial.suggest_float("tp_atr",0.5,4.0)
    pnl = run_fast(o,h,l,c,atr,fast,slow,entry,sl,tp)
    if len(pnl)<5: return -9999
    wr=(pnl>0).mean()
    sh=pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else -9
    return sh*(1+wr-0.5)*np.sign(pnl.sum()+0.01)

def evaluate_combination(dfs, train_days, n_champions):
    all_equity = []
    for tf_name, (tf_val, _, max_slow) in TF_CONFIG.items():
        df = dfs.get(tf_name)
        if df is None: continue
        start_date = df.index[0]
        end_date = df.index[-1]
        windows = []
        t_start = start_date
        while t_start + pd.Timedelta(days=train_days+TEST_DAYS) <= end_date:
            tr_s = t_start
            tr_e = t_start + pd.Timedelta(days=train_days)
            te_s = tr_e
            te_e = tr_e + pd.Timedelta(days=TEST_DAYS)
            if te_e <= end_date:
                windows.append((tr_s, tr_e, te_s, te_e))
            t_start += pd.Timedelta(days=STEP_DAYS)
        if not windows: continue
        for w, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            df_train = df[(df.index >= tr_s) & (df.index < tr_e)]
            df_test  = df[(df.index >= te_s) & (df.index < te_e)]
            if len(df_train) < 500 or len(df_test) < 50: continue
            o,h,l,c,atr = precompute(df_train)
            o2,h2,l2,c2,atr2 = precompute(df_test)
            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
            study.optimize(lambda t: objective(t,o,h,l,c,atr,max_slow),
                           n_trials=N_TRIALS, show_progress_bar=False)
            trials = sorted([t for t in study.trials if t.value is not None and t.value>-9990],
                            key=lambda t:t.value, reverse=True)
            window_pnl = []
            for t in trials[:n_champions]:
                p = t.params
                pnl_t = run_fast(o2,h2,l2,c2,atr2,**p)
                window_pnl.append(pnl_t)
            if window_pnl:
                max_len = max(len(x) for x in window_pnl)
                combined = np.zeros(max_len)
                for pnl in window_pnl:
                    combined[:len(pnl)] += pnl
                all_equity.append(combined)
    if not all_equity:
        return None
    max_total = max(len(x) for x in all_equity)
    total = np.zeros(max_total)
    for pnl in all_equity:
        total[:len(pnl)] += pnl
    n = len(total)
    wr = (total>0).mean() if n>0 else 0
    sh = total.mean()/total.std()*np.sqrt(252) if n>5 and total.std()>0 else 0
    tot = total.sum()
    cum = np.cumsum(total)
    mdd = np.max(np.maximum.accumulate(cum)-cum) if n>0 else 0
    n_windows = len(all_equity)
    profit_windows = sum(1 for p in all_equity if p.sum()>0)
    if n >= 20:
        rs = pd.Series(total).rolling(20).mean()/pd.Series(total).rolling(20).std()*np.sqrt(252)
        rs_clean = rs.dropna()
        rs_min = rs_clean.min() if len(rs_clean)>0 else 0
        rs_pct_pos = (rs_clean>0).mean() if len(rs_clean)>0 else 0
    else:
        rs_min = 0; rs_pct_pos = 0
    return {
        "train_days": train_days,
        "n_champions": n_champions,
        "total_pips": tot,
        "sharpe": sh,
        "win_rate": wr,
        "mdd": mdd,
        "n_trades": n,
        "n_windows": n_windows,
        "profit_windows": profit_windows,
        "win_pct": profit_windows/n_windows if n_windows>0 else 0,
        "rs_min": rs_min,
        "rs_pct_positive": rs_pct_pos,
        "equity": total,
    }

if __name__ == "__main__":
    try: mt5.shutdown()
    except: pass

    print("="*70)
    print("MA V9 - Meta-Optimization EURUSD (CORRECTED ENTRY o[i+1])")
    print(f"Grid: champions={CHAMPION_COUNTS} x train={TRAIN_DAYS_LIST}d")
    print(f"Test={TEST_DAYS}d | cost={COST_PER_TRADE} pips/trade | trials={N_TRIALS}")
    print("="*70)

    print("\nLoading data...")
    dfs = {}
    for tf_name, (tf_val, n_bars, _) in TF_CONFIG.items():
        raw = load_data(tf_val, n_bars)
        if raw is not None:
            dfs[tf_name] = raw.dropna()
            print(f"  {tf_name}: {len(dfs[tf_name])} bars")

    results = []
    total_combos = len(CHAMPION_COUNTS) * len(TRAIN_DAYS_LIST)
    combo_num = 0

    for train_days in TRAIN_DAYS_LIST:
        for n_champ in CHAMPION_COUNTS:
            combo_num += 1
            t0 = time.time()
            print(f"[{combo_num}/{total_combos}] train={train_days}d champs={n_champ}...", end=" ", flush=True)
            res = evaluate_combination(dfs, train_days, n_champ)
            dt = time.time()-t0
            if res:
                results.append(res)
                rdf_tmp = pd.DataFrame([{k:v for k,v in r.items() if k!="equity"} for r in results])
                rdf_tmp.to_csv("D:\\PYTHON\\METATRADER\\ma_v9_grid.csv", index=False)
                print(f"pips={res['total_pips']:+.0f} sh={res['sharpe']:.1f} "
                      f"wr={res['win_rate']:.0%} mdd={res['mdd']:.0f} "
                      f"winW={res['win_pct']:.0%} ({dt:.0f}s)")
            else:
                print(f"SKIP ({dt:.0f}s)")

    rdf = pd.DataFrame([{k:v for k,v in r.items() if k!="equity"} for r in results])
    rdf.to_csv("D:\\PYTHON\\METATRADER\\ma_v9_grid.csv", index=False)

    print(f"\n{'='*70}")
    print("RESULTS MATRIX (V9 - CORRECTED)")
    print(f"{'='*70}")

    print("\n--- SHARPE ---")
    pivot_sh = rdf.pivot_table(index="n_champions", columns="train_days", values="sharpe")
    print(pivot_sh.to_string())

    print("\n--- WIN RATE ---")
    pivot_wr = rdf.pivot_table(index="n_champions", columns="train_days", values="win_rate")
    print(pivot_wr.to_string())

    print("\n--- MDD (lower=better) ---")
    pivot_mdd = rdf.pivot_table(index="n_champions", columns="train_days", values="mdd")
    print(pivot_mdd.to_string())

    print("\n--- PROFITABLE WINDOWS % ---")
    pivot_win = rdf.pivot_table(index="n_champions", columns="train_days", values="win_pct")
    print(pivot_win.to_string())

    print(f"\n{'='*70}")
    print("TOP-5 BY SHARPE")
    print(f"{'='*70}")
    top_sh = rdf.nlargest(5, "sharpe")
    for _, r in top_sh.iterrows():
        print(f"  train={r['train_days']:.0f}d champs={r['n_champions']:.0f} | "
              f"sh={r['sharpe']:.2f} pips={r['total_pips']:+.0f} wr={r['win_rate']:.0%} "
              f"mdd={r['mdd']:.0f} winW={r['win_pct']:.0%}")

    rdf["score"] = (rdf["sharpe"] * 0.3 +
                     rdf["win_rate"] * 100 * 0.3 -
                     rdf["mdd"] * 0.2 +
                     rdf["win_pct"] * 100 * 0.2)
    print(f"\n{'='*70}")
    print("TOP-5 BALANCED")
    print(f"{'='*70}")
    top_bal = rdf.nlargest(5, "score")
    for _, r in top_bal.iterrows():
        print(f"  train={r['train_days']:.0f}d champs={r['n_champions']:.0f} | "
              f"score={r['score']:.1f} sh={r['sharpe']:.2f} wr={r['win_rate']:.0%} "
              f"mdd={r['mdd']:.0f} pips={r['total_pips']:+.0f} winW={r['win_pct']:.0%}")

    # Save best params for MQL5
    best = rdf.nlargest(1, "score").iloc[0]
    best_train = int(best["train_days"])
    best_n = int(best["n_champions"])
    print(f"\n{'='*70}")
    print(f"BEST: train={best_train}d champions={best_n}")
    print(f"Extracting champion parameters for MQL5...")
    print(f"{'='*70}")

    for tf_name, (tf_val, _, max_slow) in TF_CONFIG.items():
        df = dfs.get(tf_name)
        if df is None: continue
        split = df.index[-1] - pd.Timedelta(days=TEST_DAYS)
        train_end = split - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=best_train)
        df_train = df[(df.index > train_start) & (df.index <= train_end)]
        if len(df_train) < 500: continue
        o,h,l,c,atr = precompute(df_train)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(lambda t: objective(t,o,h,l,c,atr,max_slow),
                       n_trials=200, show_progress_bar=False)
        trials = sorted([t for t in study.trials if t.value is not None and t.value>-9990],
                        key=lambda t:t.value, reverse=True)[:best_n]
        print(f"\n{tf_name} ({len(trials)} champions):")
        for i, t in enumerate(trials):
            p = t.params
            print(f"  #{i+1}: fast={p['fast']} slow={p['slow']} entry={p['entry']:.3f} "
                  f"sl_atr={p['sl_atr']:.2f} tp_atr={p['tp_atr']:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, col, fmt, cmap) in zip(axes.flat, [
        ("Sharpe Ratio", "sharpe", "{:.1f}", "RdYlGn"),
        ("Win Rate", "win_rate", "{:.0%}", "RdYlGn"),
        ("Max Drawdown", "mdd", "{:.0f}", "RdYlGn_r"),
        ("Profitable Windows", "win_pct", "{:.0%}", "RdYlGn"),
    ]):
        pivot = rdf.pivot_table(index="n_champions", columns="train_days", values=col)
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(x)}d" for x in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([int(x) for x in pivot.index])
        ax.set_xlabel("Train Days")
        ax.set_ylabel("N Champions")
        ax.set_title(title)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle("EURUSD MA V9 - Corrected Entry (o[i+1])", fontsize=14)
    plt.tight_layout()
    plt.savefig("D:\\PYTHON\\METATRADER\\ma_v9_heatmaps.png", dpi=150)
    print(f"\nSaved: ma_v9_heatmaps.png + ma_v9_grid.csv")
