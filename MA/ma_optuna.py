"""
MA crossover + Optuna + top-5 OOS + equity plot
EURUSD H1 | train: 1yr, test: 1mo | fast version
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
TIMEFRAME = mt5.TIMEFRAME_H1
PIP       = 0.0001

def load_data(n_bars=20000):
    if not mt5.initialize():
        raise RuntimeError(f"MT5: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown(); raise RuntimeError(f"No {SYMBOL}")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n_bars)
    mt5.shutdown()
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
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().values
    return o, h, l, c, atr

def run_fast(o, h, l, c, atr, fast, slow, entry, sl_atr, tp_atr):
    n = len(c)
    maf = pd.Series(c).rolling(fast).mean().values
    mas = pd.Series(c).rolling(slow).mean().values
    denom = np.where(atr > 1e-10, atr, 1e-10)
    diff = (maf - mas) / denom

    trades = []
    pos = 0; ep = sl = tp = 0.0; ei = 0
    s = max(fast, slow, 14)
    for i in range(s, n):
        d = diff[i]
        if pos == 0:
            if np.isnan(d): continue
            if d > entry:
                pos = 1; ep = o[i]; sl = ep - atr[i]*sl_atr; tp = ep + atr[i]*tp_atr; ei = i
            elif d < -entry:
                pos = -1; ep = o[i]; sl = ep + atr[i]*sl_atr; tp = ep - atr[i]*tp_atr; ei = i
        else:
            b = i - ei
            if pos == 1:
                if   l[i] <= sl: trades.append(-(sl-ep)/PIP); pos=0
                elif h[i] >= tp: trades.append((tp-ep)/PIP);  pos=0
                elif d < -entry*0.3 or (b>=3 and d<=0): trades.append((c[i]-ep)/PIP); pos=0
            else:
                if   h[i] >= sl: trades.append((ep-sl)/PIP);  pos=0
                elif l[i] <= tp: trades.append((ep-tp)/PIP);  pos=0
                elif d > entry*0.3 or (b>=3 and d>=0): trades.append((ep-c[i])/PIP); pos=0
    return np.array(trades) if trades else np.array([])

def stats(pnl):
    if len(pnl) < 5: return -9999
    wr = (pnl>0).mean()
    sh = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else -9
    return sh * (1 + wr - 0.5) * np.sign(pnl.sum()+0.01)

def objective(trial, o, h, l, c, atr):
    fast = trial.suggest_int("fast", 5, 100)
    slow = trial.suggest_int("slow", fast+15, 250)
    entry = trial.suggest_float("entry", 0.1, 1.5)
    sl = trial.suggest_float("sl_atr", 0.5, 4.0)
    tp = trial.suggest_float("tp_atr", 0.5, 4.0)
    pnl = run_fast(o, h, l, c, atr, fast, slow, entry, sl, tp)
    return stats(pnl)

if __name__ == "__main__":
    print("Loading EURUSD H1...")
    raw = load_data(20000)
    df = raw.dropna()
    print(f"Total: {len(df)} bars | {df.index[0].date()} -> {df.index[-1].date()}")

    split = df.index[-1] - pd.Timedelta(days=30)
    train_end = split - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=365)
    df_train = df[(df.index > train_start) & (df.index <= train_end)]
    df_test  = df[df.index > split]
    print(f"Train: {len(df_train)} bars ({train_start.date()} -> {train_end.date()})")
    print(f"Test:  {len(df_test)} bars ({df_test.index[0].date()} -> {df_test.index[-1].date()})")

    o,h,l,c,atr = precompute(df_train)
    o2,h2,l2,c2,atr2 = precompute(df_test)

    print("\n-- Optuna 100 trials --")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(lambda t: objective(t, o, h, l, c, atr), n_trials=100)
    dt = (time.time()-t0)
    print(f"Done: {dt:.0f}s ({dt/100:.1f}s/trial)")

    trials = sorted([t for t in study.trials if t.value is not None and t.value > -9990],
                    key=lambda t: t.value, reverse=True)
    top5 = [t.params for t in trials[:5]]

    print(f"\nTop-5:")
    for i, p in enumerate(top5):
        print(f"  #{i+1}: {p}")

    print(f"\n-- OOS Test --")
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    for i, p in enumerate(top5):
        pnl = run_fast(o2, h2, l2, c2, atr2, **p)
        n = len(pnl)
        wr = (pnl>0).mean()
        sh = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else 0
        tot = pnl.sum()
        cum = np.cumsum(pnl)
        mdd = np.max(np.maximum.accumulate(cum)-cum) if n>0 else 0
        lbl = f"#{i+1} f={p['fast']} s={p['slow']} sh={sh:.2f} pips={tot:.0f}"
        ax.plot(cum, label=lbl, color=colors[i], lw=1.5)
        print(f"  #{i+1}: fast={p['fast']} slow={p['slow']} entry={p['entry']:.2f} sl={p['sl_atr']:.1f} tp={p['tp_atr']:.1f}")
        print(f"       n={n} wr={wr:.0%} sharpe={sh:.3f} pips={tot:.0f} mdd={mdd:.0f}")

    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative Pips")
    ax.set_title(f"Top-5 MA | OOS {df_test.index[0].date()} to {df_test.index[-1].date()}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("D:\\PYTHON\\METATRADER\\ma_oos_equity.png", dpi=150)
    print(f"\nSaved: ma_oos_equity.png")
