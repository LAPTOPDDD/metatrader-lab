"""
MA Ensemble - CORRECTED Python (no lookahead)
Optimize + backtest + equity curves
Can trade live via MT5 Python API
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, time, json
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SYMBOL = "EURUSD"
PIP = 0.0001
COST = 0.7
LOT = 0.01
TF_CONFIG = {
    "M15": (mt5.TIMEFRAME_M15, 30000, 50),
    "M30": (mt5.TIMEFRAME_M30, 20000, 60),
    "H1":  (mt5.TIMEFRAME_H1,  20000, 80),
}
N_TRIALS = 200
N_CHAMPIONS = 5
TRAIN_DAYS = 545
TEST_DAYS = 30

# ============================================================
# DATA
# ============================================================
def load_data(tf, n_bars):
    try: mt5.shutdown()
    except: pass
    mt5.initialize(); mt5.symbol_select(SYMBOL, True)
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n_bars)
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
    pc = np.roll(c,1); pc[0]=c[0]
    tr = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc)))
    atr = pd.Series(tr).rolling(14).mean().values
    return o, h, l, c, atr

# ============================================================
# STRATEGY (corrected: entry at o[i+1])
# ============================================================
def run_strategy(o, h, l, c, atr, fast, slow, entry_thr, sl_atr, tp_atr, cooldown=3):
    n = len(c)
    maf = pd.Series(c).rolling(fast).mean().values
    mas = pd.Series(c).rolling(slow).mean().values
    denom = np.where(atr > 1e-10, atr, 1e-10)
    diff = (maf - mas) / denom
    
    pos = 0; ep = sl_v = tp_v = 0.0; ei = 0
    trades = []; equities = [0.0]; bal = 0.0
    last_sl_bar = -999
    s = max(fast, slow, 14)
    
    for i in range(s, n):
        d = diff[i]
        if pos == 0:
            if np.isnan(d): 
                equities.append(bal)
                continue
            cooled = (last_sl_bar >= 0 and (i - last_sl_bar) < cooldown)
            if d > entry_thr and (i+1) < n and not cooled:
                pos = 1; ep = o[i+1]; sl_v = ep - atr[i]*sl_atr; tp_v = ep + atr[i]*tp_atr; ei = i+1
            elif d < -entry_thr and (i+1) < n and not cooled:
                pos = -1; ep = o[i+1]; sl_v = ep + atr[i]*sl_atr; tp_v = ep - atr[i]*tp_atr; ei = i+1
        else:
            b = i - ei
            pnl = 0
            if pos == 1:
                if l[i] <= sl_v: pnl = -(sl_v - ep)/PIP - COST; pos = 0; last_sl_bar = i
                elif h[i] >= tp_v: pnl = (tp_v - ep)/PIP - COST; pos = 0
                elif d > -entry_thr*0.3 or (b >= 3 and d <= 0): pnl = (c[i] - ep)/PIP - COST; pos = 0
            else:
                if h[i] >= sl_v: pnl = (ep - sl_v)/PIP - COST; pos = 0; last_sl_bar = i
                elif l[i] <= tp_v: pnl = (ep - tp_v)/PIP - COST; pos = 0
                elif d < entry_thr*0.3 or (b >= 3 and d >= 0): pnl = (ep - c[i])/PIP - COST; pos = 0
            if pnl != 0:
                trades.append(pnl)
                bal += pnl * LOT * 10
        equities.append(bal)
    
    return np.array(trades), np.array(equities)

def objective(trial, o, h, l, c, atr, max_slow):
    fast = trial.suggest_int("fast", 3, max(3, max_slow-11))
    slow = trial.suggest_int("slow", fast+10, max_slow)
    entry = trial.suggest_float("entry", 0.05, 1.2)
    sl = trial.suggest_float("sl_atr", 0.5, 4.0)
    tp = trial.suggest_float("tp_atr", 0.5, 4.0)
    cd = trial.suggest_int("cooldown", 1, 8)
    trades, _ = run_strategy(o, h, l, c, atr, fast, slow, entry, sl, tp, cd)
    if len(trades) < 5: return -9999
    wr = (trades > 0).mean()
    sh = trades.mean() / trades.std() * np.sqrt(252) if trades.std() > 0 else -9
    return sh * (1 + wr - 0.5) * np.sign(trades.sum() + 0.01)

# ============================================================
# OPTIMIZE + EXTRACT CHAMPIONS
# ============================================================
def optimize_tf(df, tf_name, max_slow):
    split = df.index[-1] - pd.Timedelta(days=TEST_DAYS)
    train_end = split - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
    df_train = df[(df.index > train_start) & (df.index <= train_end)]
    if len(df_train) < 500:
        print(f"  {tf_name}: not enough data ({len(df_train)} bars)")
        return []
    
    o, h, l, c, atr = precompute(df_train)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(lambda t: objective(t, o, h, l, c, atr, max_slow), n_trials=N_TRIALS, show_progress_bar=False)
    
    trials = sorted([t for t in study.trials if t.value is not None and t.value > -9990],
                    key=lambda t: t.value, reverse=True)[:N_CHAMPIONS]
    
    champions = []
    for t in trials:
        p = t.params
        champions.append({
            "fast": p["fast"], "slow": p["slow"], "entry": p["entry"],
            "sl_atr": p["sl_atr"], "tp_atr": p["tp_atr"], "cooldown": p["cooldown"]
        })
    return champions

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try: mt5.shutdown()
    except: pass
    
    print("="*70)
    print("MA Ensemble - CORRECTED Python (no lookahead)")
    print(f"Train={TRAIN_DAYS}d Test={TEST_DAYS}d | Champions={N_CHAMPIONS}")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    dfs = {}
    for tf_name, (tf_val, n_bars, _) in TF_CONFIG.items():
        raw = load_data(tf_val, n_bars)
        if raw is not None:
            dfs[tf_name] = raw.dropna()
            print(f"  {tf_name}: {len(dfs[tf_name])} bars")
    
    # Optimize each TF
    print("\nOptimizing...")
    all_champions = {}
    for tf_name, (_, _, max_slow) in TF_CONFIG.items():
        print(f"\n{tf_name}:")
        champs = optimize_tf(dfs[tf_name], tf_name, max_slow)
        all_champions[tf_name] = champs
        for i, ch in enumerate(champs):
            print(f"  #{i+1}: fast={ch['fast']} slow={ch['slow']} entry={ch['entry']:.3f} "
                  f"sl={ch['sl_atr']:.2f} tp={ch['tp_atr']:.2f} cd={ch['cooldown']}")
    
    # Save champions
    with open("D:\\PYTHON\\METATRADER\\champions_corrected.json", "w") as fp:
        json.dump(all_champions, fp, indent=2)
    
    # ============================================================
    # BACKTEST on TEST period
    # ============================================================
    print("\n" + "="*70)
    print("TEST PERIOD (last 30 days)")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    all_equity = []
    grand_p = 0; grand_t = 0; grand_w = 0
    
    for tf_idx, (tf_name, (_, _, _)) in enumerate(TF_CONFIG.items()):
        df = dfs[tf_name]
        split = df.index[-1] - pd.Timedelta(days=TEST_DAYS)
        test = df[df.index >= split]
        o, h, l, c, atr = precompute(test)
        
        tf_p = 0; tf_t = 0; tf_w = 0
        tf_eq = np.zeros(len(test))
        
        ax = axes.flat[tf_idx]
        
        for i, ch in enumerate(all_champions[tf_name]):
            trades, eq = run_strategy(o, h, l, c, atr, ch["fast"], ch["slow"],
                                       ch["entry"], ch["sl_atr"], ch["tp_atr"], ch["cooldown"])
            p = sum(trades)
            tf_p += p; tf_t += len(trades); tf_w += (trades > 0).sum()
            
            # Equity curve
            eq_usd = eq * LOT * 10
            ax.plot(eq_usd, label=f"Ch{i+1} ({p:+.0f}p)", linewidth=0.8)
            
            # Add to combined
            min_len = min(len(tf_eq), len(eq_usd))
            tf_eq[:min_len] += eq_usd[:min_len]
        
        wr = tf_w/tf_t*100 if tf_t > 0 else 0
        print(f"\n{tf_name}: {tf_p:+.1f}p (${tf_p*LOT*10:+.2f}) | {tf_t} trades | WR {wr:.0f}%")
        
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{tf_name}: {tf_p:+.1f}p | {tf_t} trades | WR {wr:.0f}%")
        ax.set_ylabel("PnL ($)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        all_equity.append(tf_eq)
        grand_p += tf_p; grand_t += tf_t; grand_w += tf_w
    
    # Combined equity
    max_len = max(len(e) for e in all_equity)
    combined = np.zeros(max_len)
    for eq in all_equity:
        combined[:len(eq)] += eq[:len(eq)]
    
    ax = axes.flat[3]
    ax.plot(combined, "b-", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    gwr = grand_w/grand_t*100 if grand_t > 0 else 0
    ax.set_title(f"COMBINED: {grand_p:+.1f}p (${grand_p*LOT*10:+.2f}) | WR {gwr:.0f}%")
    ax.set_ylabel("PnL ($)")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("V9 Corrected - Test Period (30 days)", fontsize=14)
    plt.tight_layout()
    plt.savefig("D:\\PYTHON\\METATRADER\\ma_v9_corrected_test.png", dpi=150)
    
    print(f"\n{'='*70}")
    print(f"GRAND TOTAL: {grand_p:+.1f}p (${grand_p*LOT*10:+.2f}) | {grand_t} trades | WR {gwr:.0f}%")
    print(f"{'='*70}")
    print(f"\nSaved: ma_v9_corrected_test.png + champions_corrected.json")
