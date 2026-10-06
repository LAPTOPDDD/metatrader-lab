"""
MA V6 - Walk-Forward with REAL costs (commission + spread from MT5)
M5 + M15 + M30 + H1 | top-5 each | realistic PnL
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
TIMEFRAMES = {
    "M5":  (mt5.TIMEFRAME_M5,  50000, 40),
    "M15": (mt5.TIMEFRAME_M15, 30000, 50),
    "M30": (mt5.TIMEFRAME_M30, 20000, 60),
    "H1":  (mt5.TIMEFRAME_H1,  20000, 80),
}
N_TRIALS   = 80
TRAIN_DAYS = 60
TEST_DAYS  = 30
STEP_DAYS  = 30
SLIPPAGE   = 1  # 1 point slippage each side

def get_broker_costs(df=None):
    """Get real costs: spread from data + commission from account"""
    try: mt5.shutdown()
    except: pass
    if not mt5.initialize():
        raise RuntimeError(f"MT5: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown(); raise RuntimeError(f"No {SYMBOL}")

    info = mt5.symbol_info(SYMBOL)
    acc = mt5.account_info()
    mt5.shutdown()

    if info is None:
        raise RuntimeError("Cannot get symbol info")

    spread_points = info.spread  # current spread in points
    point = info.point  # 1e-05 for EURUSD

    # Try to get commission from account
    # MT5 doesn't expose commission directly, use typical broker values
    # For raw accounts: ~$3-7 per lot RT = ~0.3-0.7 pips for EURUSD
    # For standard accounts: 0 commission but wider spread
    # We'll use a conservative estimate
    COMMISSION_PIPS = 0.3  # ~$3 per lot RT (conservative)

    spread_pips = spread_points * point / PIP  # convert points to pips

    return {
        "spread_points": spread_points,
        "spread_pips": spread_pips,
        "commission_pips": COMMISSION_PIPS,
        "point": point,
    }

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
        raise RuntimeError(f"No data for tf={tf}")
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

def run_fast(o, h, l, c, atr, fast, slow, entry, sl_atr, tp_atr,
             cost_per_trade=0.0):
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
            if d > entry:
                pos=1; ep=o[i]; sl=ep-atr[i]*sl_atr; tp=ep+atr[i]*tp_atr; ei=i
            elif d < -entry:
                pos=-1; ep=o[i]; sl=ep+atr[i]*sl_atr; tp=ep-atr[i]*tp_atr; ei=i
        else:
            b = i-ei
            if pos==1:
                if   l[i]<=sl: trades.append(-(sl-ep)/PIP - cost_per_trade); pos=0
                elif h[i]>=tp: trades.append((tp-ep)/PIP - cost_per_trade);  pos=0
                elif d<-entry*0.3 or (b>=3 and d<=0): trades.append((c[i]-ep)/PIP - cost_per_trade); pos=0
            else:
                if   h[i]>=sl: trades.append((ep-sl)/PIP - cost_per_trade);  pos=0
                elif l[i]<=tp: trades.append((ep-tp)/PIP - cost_per_trade);  pos=0
                elif d>entry*0.3 or (b>=3 and d>=0): trades.append((ep-c[i])/PIP - cost_per_trade); pos=0
    return np.array(trades) if trades else np.array([])

def objective(trial, o, h, l, c, atr, max_slow, cost):
    fast = trial.suggest_int("fast",3,max(3,max_slow-11))
    slow = trial.suggest_int("slow",fast+10,max_slow)
    entry = trial.suggest_float("entry",0.05,1.2)
    sl = trial.suggest_float("sl_atr",0.5,4.0)
    tp = trial.suggest_float("tp_atr",0.5,4.0)
    pnl = run_fast(o,h,l,c,atr,fast,slow,entry,sl,tp,cost)
    if len(pnl)<5: return -9999
    wr=(pnl>0).mean()
    sh=pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else -9
    return sh*(1+wr-0.5)*np.sign(pnl.sum()+0.01)

def walk_forward_single_tf(df, tf_name, max_slow, windows, cost):
    results = []
    all_oos_pnl = []

    for w, (train_start, train_end, test_start, test_end) in enumerate(windows):
        df_train = df[(df.index >= train_start) & (df.index < train_end)]
        df_test  = df[(df.index >= test_start) & (df.index < test_end)]
        if len(df_train) < 500 or len(df_test) < 50: continue

        o,h,l,c,atr = precompute(df_train)
        o2,h2,l2,c2,atr2 = precompute(df_test)

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(lambda t: objective(t,o,h,l,c,atr,max_slow,cost),
                       n_trials=N_TRIALS, show_progress_bar=False)

        trials = sorted([t for t in study.trials if t.value is not None and t.value>-9990],
                        key=lambda t:t.value, reverse=True)

        window_pnl = []
        for t in trials[:5]:
            p = t.params
            pnl_t = run_fast(o2,h2,l2,c2,atr2,**p,cost_per_trade=cost)
            window_pnl.append(pnl_t)

        max_len = max(len(x) for x in window_pnl) if window_pnl else 0
        combined = np.zeros(max_len)
        for pnl in window_pnl:
            combined[:len(pnl)] += pnl

        n = len(combined)
        wr = (combined>0).mean() if n>0 else 0
        sh = combined.mean()/combined.std()*np.sqrt(252) if n>5 and combined.std()>0 else 0
        tot = combined.sum()
        mdd = np.max(np.maximum.accumulate(np.cumsum(combined))-np.cumsum(combined)) if n>0 else 0

        results.append({
            "window": w+1, "test": f"{test_start.date()}->{test_end.date()}",
            "trades": n, "wr": wr, "sharpe": sh, "pips": tot, "mdd": mdd,
        })
        all_oos_pnl.append(combined)

    return results, all_oos_pnl

if __name__ == "__main__":
    # Get broker costs
    costs = get_broker_costs()
    SLIPPAGE_PIPS = 0.2  # 2 points = 0.2 pips each side
    cost_per_trade = costs["spread_pips"] + costs["commission_pips"] + SLIPPAGE_PIPS * 2

    print("="*70)
    print("MA V6 - Walk-Forward with REAL COSTS")
    print(f"M5 + M15 + M30 + H1 | train={TRAIN_DAYS}d test={TEST_DAYS}d")
    print("="*70)
    print(f"\nBROKER COSTS ({SYMBOL}):")
    print(f"  Spread: {costs['spread_points']} points = {costs['spread_pips']:.1f} pips")
    print(f"  Commission: ~{costs['commission_pips']:.1f} pips ($3/lot RT)")
    print(f"  Slippage: {SLIPPAGE_PIPS:.1f} pips each side")
    print(f"  TOTAL COST PER TRADE: {cost_per_trade:.1f} pips")

    dfs = {}
    for tf_name, (tf_val, n_bars, _) in TIMEFRAMES.items():
        print(f"\nLoading {tf_name}...")
        raw = load_data(tf_val, n_bars)
        df = raw.dropna()
        dfs[tf_name] = df
        print(f"  {len(df)} bars | {df.index[0].date()} -> {df.index[-1].date()}")

    all_results = {}
    all_oos = {}

    for tf_name, (_, _, max_slow) in TIMEFRAMES.items():
        df = dfs[tf_name]
        start_date = df.index[0]
        end_date = df.index[-1]
        windows = []
        t_start = start_date
        while t_start + pd.Timedelta(days=TRAIN_DAYS+TEST_DAYS) <= end_date:
            tr_s = t_start
            tr_e = t_start + pd.Timedelta(days=TRAIN_DAYS)
            te_s = tr_e
            te_e = tr_e + pd.Timedelta(days=TEST_DAYS)
            if te_e <= end_date:
                windows.append((tr_s, tr_e, te_s, te_e))
            t_start += pd.Timedelta(days=STEP_DAYS)

        print(f"\n-- {tf_name} | {len(windows)} windows | cost={cost_per_trade:.1f} pips/trade --")
        res, oos = walk_forward_single_tf(df, tf_name, max_slow, windows, cost_per_trade)
        all_results[tf_name] = res
        all_oos[tf_name] = oos

        for r in res:
            mark = "+" if r["pips"] > 0 else "-"
            print(f"  W{r['window']:>2}: {r['test']} | n={r['trades']:>3} "
                  f"wr={r['wr']:.0%} sh={r['sharpe']:>6.1f} "
                  f"pips={r['pips']:>+7.0f} mdd={r['mdd']:>5.0f} {mark}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY (costs included: {cost_per_trade:.1f} pips/trade)")
    print(f"{'='*70}")
    for tf_name in TIMEFRAMES:
        res = all_results[tf_name]
        if not res: continue
        profitable = sum(1 for r in res if r["pips"] > 0)
        total = len(res)
        avg_sh = np.mean([r["sharpe"] for r in res])
        total_pips = sum(r["pips"] for r in res)
        avg_wr = np.mean([r["wr"] for r in res])
        total_trades = sum(r["trades"] for r in res)
        total_costs = total_trades * cost_per_trade
        print(f"  {tf_name:>4}: {profitable}/{total} profitable | "
              f"avg_sh={avg_sh:.1f} pips={total_pips:>+7.0f} wr={avg_wr:.0%} "
              f"trades={total_trades} costs={total_costs:.0f}")

    # Global ensemble
    print(f"\n{'='*70}")
    print("GLOBAL ENSEMBLE")
    print(f"{'='*70}")

    min_windows = min(len(v) for v in all_results.values() if v)
    global_pnl = []
    window_stats = []

    for w in range(min_windows):
        window_combined = []
        for tf_name in TIMEFRAMES:
            if w < len(all_oos[tf_name]):
                arr = all_oos[tf_name][w]
                if isinstance(arr, np.ndarray) and len(arr) > 0:
                    window_combined.append(arr)

        if not window_combined: continue
        max_len = max(len(x) for x in window_combined)
        ens = np.zeros(max_len)
        for pnl in window_combined:
            ens[:len(pnl)] += pnl

        n = len(ens)
        wr = (ens>0).mean() if n>0 else 0
        sh = ens.mean()/ens.std()*np.sqrt(252) if n>5 and ens.std()>0 else 0
        tot = ens.sum()
        global_pnl.append(ens)
        window_stats.append({"w":w+1,"n":n,"wr":wr,"sh":sh,"pips":tot})
        mark = "+" if tot > 0 else "-"
        print(f"  W{w+1:>2}: n={n:>3} wr={wr:.0%} sh={sh:>6.1f} pips={tot:>+7.0f} {mark}")

    if global_pnl:
        max_total = max(len(x) for x in global_pnl)
        total_ens = np.zeros(max_total)
        for pnl in global_pnl:
            total_ens[:len(pnl)] += pnl

        tn = len(total_ens)
        twr = (total_ens>0).mean() if tn>0 else 0
        tsh = total_ens.mean()/total_ens.std()*np.sqrt(252) if tn>5 and total_ens.std()>0 else 0
        ttot = total_ens.sum()
        tmdd = np.max(np.maximum.accumulate(np.cumsum(total_ens))-np.cumsum(total_ens))
        profitable_w = sum(1 for s in window_stats if s["pips"]>0)
        total_w = len(window_stats)

        print(f"\n  TOTAL: {profitable_w}/{total_w} profitable windows")
        print(f"  TOTAL PIPS (after costs): {ttot:>+7.0f}")
        print(f"  TOTAL SHARPE: {tsh:.2f}")
        print(f"  TOTAL MDD: {tmdd:.0f}")
        print(f"  AVG WIN RATE: {twr:.0%}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    colors = {"M5":"#e41a1c","M15":"#ff7f00","M30":"#4daf4a","H1":"#377eb8"}

    ax = axes[0]
    for tf_name in TIMEFRAMES:
        for i, pnl in enumerate(all_oos.get(tf_name, [])):
            cum = np.cumsum(pnl)
            ax.plot(cum, color=colors[tf_name], alpha=0.4, lw=1)
    for tf, c in colors.items():
        ax.plot([], [], color=c, label=tf, lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("Pips (after costs)")
    ax.set_title(f"OOS Equity (costs={cost_per_trade:.1f} pips/trade: spread+comm+slippage)")
    ax.legend(fontsize=9, ncol=4); ax.grid(True, alpha=0.3)

    ax = axes[1]
    wdf = pd.DataFrame(window_stats)
    bar_colors = ["#2ca02c" if x>0 else "#d62728" for x in wdf["pips"]]
    ax.bar(wdf["w"], wdf["pips"], color=bar_colors, alpha=0.7)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Window #"); ax.set_ylabel("Pips (after costs)")
    ax.set_title("Pips per Window (after all costs)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("D:\\PYTHON\\METATRADER\\ma_v6_costs.png", dpi=150)
    print(f"\nSaved: ma_v6_costs.png")
