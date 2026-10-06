"""
MA crossover V3 - Trend-filtered ensemble
H1 + H4 | top-5 each | trend + volatility filters | 60-day OOS
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
    "H1":  (mt5.TIMEFRAME_H1,  8000),
    "H4":  (mt5.TIMEFRAME_H4,  8000),
}
N_TRIALS  = 200
TEST_DAYS = 60

def load_data(tf, n_bars):
    if not mt5.initialize():
        raise RuntimeError(f"MT5: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown(); raise RuntimeError(f"No {SYMBOL}")
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n_bars)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df[df.index.dayofweek < 5]
    return df[["open","high","low","close"]].copy()

def add_filters(df):
    df = df.copy()
    # ATR
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"]-df["low"],
        (df["high"]-prev_c).abs(),
        (df["low"]-prev_c).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Trend: price vs 200 MA
    df["ma200"] = df["close"].rolling(200).mean()
    df["trend"] = np.where(df["close"] > df["ma200"], 1, -1)

    # Volatility: ATR percentile (low vol = good for MA)
    df["atr_pct"] = df["atr"].rolling(100).rank(pct=True)

    # Momentum: rate of change
    df["roc"] = (df["close"] - df["close"].shift(20)) / df["close"].shift(20)

    return df.dropna()

def precompute(df):
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    atr = df["atr"].values.astype(np.float64)
    trend = df["trend"].values.astype(np.float64)
    atr_pct = df["atr_pct"].values.astype(np.float64)
    roc = df["roc"].values.astype(np.float64)
    return o, h, l, c, atr, trend, atr_pct, roc

def run_v3(o, h, l, c, atr, trend, atr_pct, roc,
           fast, slow, entry, sl_atr, tp_atr,
           use_trend, use_vol, vol_lo, vol_hi, use_mom):
    n = len(c)
    maf = pd.Series(c).rolling(fast).mean().values
    mas = pd.Series(c).rolling(slow).mean().values
    denom = np.where(atr>1e-10, atr, 1e-10)
    diff = (maf - mas) / denom

    trades = []
    pos=0; ep=sl=tp=0.0; ei=0
    s = max(fast, slow, 14)+50  # warmup for ma200 etc
    for i in range(s, n):
        d = diff[i]
        if pos==0:
            if np.isnan(d): continue
            # Trend filter
            if use_trend and trend[i] <= 0 and d > 0: continue
            if use_trend and trend[i] >= 0 and d < 0: continue
            # Volatility filter
            if use_vol and not (vol_lo <= atr_pct[i] <= vol_hi): continue
            # Momentum filter
            if use_mom and abs(roc[i]) < 0.005: continue  # skip flat

            if d > entry:
                pos=1; ep=o[i]; sl=ep-atr[i]*sl_atr; tp=ep+atr[i]*tp_atr; ei=i
            elif d < -entry:
                pos=-1; ep=o[i]; sl=ep+atr[i]*sl_atr; tp=ep-atr[i]*tp_atr; ei=i
        else:
            b = i-ei
            if pos==1:
                if   l[i]<=sl: trades.append(-(sl-ep)/PIP); pos=0
                elif h[i]>=tp: trades.append((tp-ep)/PIP);  pos=0
                elif d<-entry*0.3 or (b>=3 and d<=0): trades.append((c[i]-ep)/PIP); pos=0
            else:
                if   h[i]>=sl: trades.append((ep-sl)/PIP);  pos=0
                elif l[i]<=tp: trades.append((ep-tp)/PIP);  pos=0
                elif d>entry*0.3 or (b>=3 and d>=0): trades.append((ep-c[i])/PIP); pos=0
    return np.array(trades) if trades else np.array([])

def objective(trial, o, h, l, c, atr, trend, atr_pct, roc, max_slow):
    fast = trial.suggest_int("fast", 3, max(3,max_slow-11))
    slow = trial.suggest_int("slow", fast+10, max_slow)
    entry = trial.suggest_float("entry", 0.05, 1.2)
    sl = trial.suggest_float("sl_atr", 0.5, 4.0)
    tp = trial.suggest_float("tp_atr", 0.5, 4.0)
    use_trend = trial.suggest_categorical("use_trend", [True, False])
    use_vol = trial.suggest_categorical("use_vol", [True, False])
    vol_lo = trial.suggest_float("vol_lo", 0.1, 0.4) if use_vol else 0.0
    vol_hi = trial.suggest_float("vol_hi", 0.6, 0.9) if use_vol else 1.0
    use_mom = trial.suggest_categorical("use_mom", [True, False])

    pnl = run_v3(o,h,l,c,atr,trend,atr_pct,roc,
                  fast,slow,entry,sl,tp,use_trend,use_vol,vol_lo,vol_hi,use_mom)
    if len(pnl) < 5: return -9999
    wr = (pnl>0).mean()
    sh = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else -9
    return sh*(1+wr-0.5)*np.sign(pnl.sum()+0.01)

if __name__ == "__main__":
    print("="*65)
    print("MA V3 - Trend-Filtered Ensemble")
    print(f"H1 + H4 | top-5 each = 10 strategies | test {TEST_DAYS} days")
    print("="*65)

    dfs = {}
    for tf_name, (tf_val, n_bars) in TIMEFRAMES.items():
        print(f"\nLoading {tf_name}...")
        raw = load_data(tf_val, n_bars)
        df = add_filters(raw)
        dfs[tf_name] = df
        print(f"  {len(df)} bars | {df.index[0].date()} -> {df.index[-1].date()}")

    min_end = min(df.index[-1] for df in dfs.values())
    split = min_end - pd.Timedelta(days=TEST_DAYS)
    train_end = split - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=365)

    all_strategies = []

    for tf_name, (_, _) in TIMEFRAMES.items():
        df = dfs[tf_name]
        df_train = df[(df.index > train_start) & (df.index <= train_end)]
        df_test  = df[df.index > split]
        max_slow = 100 if tf_name=="H1" else 45
        print(f"\n-- {tf_name} | train: {len(df_train)} | test: {len(df_test)} --")

        o,h,l,c,atr,trend,atr_pct,roc = precompute(df_train)
        o2,h2,l2,c2,atr2,trend2,atr_pct2,roc2 = precompute(df_test)

        t0=time.time()
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(lambda t: objective(t,o,h,l,c,atr,trend,atr_pct,roc,max_slow),
                       n_trials=N_TRIALS)
        dt=time.time()-t0
        print(f"  {dt:.0f}s ({dt/N_TRIALS:.1f}s/trial)")

        trials = sorted([t for t in study.trials if t.value is not None and t.value>-9990],
                        key=lambda t:t.value, reverse=True)

        for i, t in enumerate(trials[:5]):
            p = t.params
            pnl = run_v3(o,h,l,c,atr,trend,atr_pct,roc,**p)
            pnl_t = run_v3(o2,h2,l2,c2,atr2,trend2,atr_pct2,roc2,**p)
            n_t = len(pnl_t)
            wr_t = (pnl_t>0).mean() if n_t>0 else 0
            sh_t = pnl_t.mean()/pnl_t.std()*np.sqrt(252) if n_t>5 and pnl_t.std()>0 else 0
            pips_t = pnl_t.sum() if n_t>0 else 0
            print(f"  #{i+1}: f={p['fast']:>2} s={p['slow']:>3} e={p['entry']:.2f} "
                  f"sl={p['sl_atr']:.1f} tp={p['tp_atr']:.1f} "
                  f"T={p['use_trend']} V={p['use_vol']} M={p['use_mom']}"
                  f"| test: n={n_t:>2} wr={wr_t:.0%} sh={sh_t:.1f} pips={pips_t:+.0f}")
            all_strategies.append({
                "tf": tf_name, "rank": i+1, "params": p, "test_pnl": pnl_t
            })

    # Ensemble
    print(f"\n{'='*65}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*65}")

    max_len = max(len(s["test_pnl"]) for s in all_strategies) if all_strategies else 0
    ensemble = np.zeros(max_len)
    for s in all_strategies:
        pnl = s["test_pnl"]
        ensemble[:len(pnl)] += pnl

    # Table
    print(f"\n{'#':>2} {'TF':>3} {'F':>3} {'S':>3} {'E':>5} {'SL':>4} {'TP':>4} {'T':>1} {'V':>1} {'M':>1} | {'N':>3} {'WR':>5} {'Sharpe':>7} {'Pips':>7}")
    print("-"*75)
    for i,s in enumerate(all_strategies):
        p = s["params"]
        pnl = s["test_pnl"]
        n = len(pnl)
        wr = (pnl>0).mean() if n>0 else 0
        sh = pnl.mean()/pnl.std()*np.sqrt(252) if n>5 and pnl.std()>0 else 0
        tot = pnl.sum()
        t_flag = "T" if p["use_trend"] else "-"
        v_flag = "V" if p["use_vol"] else "-"
        m_flag = "M" if p["use_mom"] else "-"
        print(f"{i+1:>2} {s['tf']:>3} {p['fast']:>3} {p['slow']:>3} "
              f"{p['entry']:.2f} {p['sl_atr']:.1f} {p['tp_atr']:.1f} "
              f"{t_flag:>1} {v_flag:>1} {m_flag:>1} | "
              f"{n:>3} {wr:>4.0%} {sh:>7.1f} {tot:>+7.0f}")

    en_n = len(ensemble)
    en_wr = (ensemble>0).mean() if en_n>0 else 0
    en_sh = ensemble.mean()/ensemble.std()*np.sqrt(252) if en_n>5 and ensemble.std()>0 else 0
    en_tot = ensemble.sum()
    en_cum = np.cumsum(ensemble)
    en_mdd = np.max(np.maximum.accumulate(en_cum)-en_cum) if en_n>0 else 0
    print("-"*75)
    print(f"{'EN':>2} {'ALL':>3} {'':>3} {'':>3} {'':>5} {'':>4} {'':>4} {'':>1} {'':>1} {'':>1} | "
          f"{en_n:>3} {en_wr:>4.0%} {en_sh:>7.1f} {en_tot:>+7.0f}  MDD={en_mdd:.0f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    colors = {"H1":"#1f77b4","H4":"#ff7f0e"}

    ax = axes[0]
    for s in all_strategies:
        cum = np.cumsum(s["test_pnl"])
        c = colors[s["tf"]]
        style = "-" if s["params"]["use_trend"] else "--"
        ax.plot(cum, color=c, alpha=0.6, lw=1.2, ls=style)
    ax.plot([], [], color=colors["H1"], label="H1", lw=2)
    ax.plot([], [], color=colors["H4"], label="H4", lw=2)
    ax.plot([], [], color="gray", label="solid=trend", lw=1, ls="-")
    ax.plot([], [], color="gray", label="dashed=no trend", lw=1, ls="--")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("Pips"); ax.set_title("Individual Strategies (T=filter, V=vol, M=mom)")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(en_cum, color="#d62728", lw=2)
    ax.fill_between(range(len(en_cum)), en_cum, alpha=0.15, color="#d62728")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("Cumulative Pips")
    ax.set_title(f"ENSEMBLE | Sharpe={en_sh:.2f} | Pips={en_tot:.0f} | MDD={en_mdd:.0f}")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if en_n >= 20:
        rs = pd.Series(ensemble).rolling(20).mean()/pd.Series(ensemble).rolling(20).std()*np.sqrt(252)
        ax.plot(rs, color="#9467bd", lw=1.5)
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.axhline(2, color="green", ls=":", lw=0.8, alpha=0.5)
    ax.set_ylabel("Sharpe"); ax.set_xlabel("Trade #"); ax.set_title("Rolling Sharpe (20)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("D:\\PYTHON\\METATRADER\\ma_v3_ensemble.png", dpi=150)
    print(f"\nSaved: ma_v3_ensemble.png")
