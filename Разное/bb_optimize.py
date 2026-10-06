"""
Bollinger Bands Ensemble - Champion Optimization
Same methodology as MA/RSI
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
PIPIP = 0.0001
COST = 0.7
LOT = 0.01
TF_CONFIG = {
    "M15": (mt5.TIMEFRAME_M15, 30000, 30),
    "M30": (mt5.TIMEFRAME_M30, 20000, 30),
    "H1":  (mt5.TIMEFRAME_H1,  20000, 30),
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

def compute_bollinger(c, period, num_std):
    sma = pd.Series(c).rolling(period).mean().values
    std = pd.Series(c).rolling(period).std().values
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower

# ============================================================
# STRATEGY: Bollinger Bands (corrected: entry at o[i+1])
# ============================================================
def run_strategy(o, h, l, c, atr, sma, upper, lower, period,
                 sl_atr, tp_atr, cooldown=3):
    n = len(c)
    pos = 0; ep = sl_v = tp_v = 0.0; ei = 0
    trades = []; equities = [0.0]; bal = 0.0
    last_sl_bar = -999
    s = period + 14
    
    for i in range(s, n):
        if pos == 0:
            if np.isnan(sma[i]): 
                equities.append(bal)
                continue
            cooled = (last_sl_bar >= 0 and (i - last_sl_bar) < cooldown)
            
            # Buy: price crosses above lower band (bounce from oversold)
            if c[i-1] < lower[i-1] and c[i] > lower[i] and (i+1) < n and not cooled:
                pos = 1; ep = o[i+1]; sl_v = ep - atr[i]*sl_atr; tp_v = ep + atr[i]*tp_atr; ei = i+1
            # Sell: price crosses below upper band (rejection from overbought)
            elif c[i-1] > upper[i-1] and c[i] < upper[i] and (i+1) < n and not cooled:
                pos = -1; ep = o[i+1]; sl_v = ep + atr[i]*sl_atr; tp_v = ep - atr[i]*tp_atr; ei = i+1
        else:
            b = i - ei
            pnl = 0
            if pos == 1:
                if l[i] <= sl_v: pnl = -(sl_v - ep)/PIPIP - COST; pos = 0; last_sl_bar = i
                elif h[i] >= tp_v: pnl = (tp_v - ep)/PIPIP - COST; pos = 0
                # Exit if price crosses above upper band (target reached)
                elif c[i] > upper[i] or (b >= 3 and c[i] < sma[i]): pnl = (c[i] - ep)/PIPIP - COST; pos = 0
            else:
                if h[i] >= sl_v: pnl = (ep - sl_v)/PIPIP - COST; pos = 0; last_sl_bar = i
                elif l[i] <= tp_v: pnl = (ep - tp_v)/PIPIP - COST; pos = 0
                elif c[i] < lower[i] or (b >= 3 and c[i] > sma[i]): pnl = (ep - c[i])/PIPIP - COST; pos = 0
            if pnl != 0:
                trades.append(pnl)
                bal += pnl * LOT * 10
        equities.append(bal)
    
    return np.array(trades), np.array(equities)

def objective(trial, o, h, l, c, atr, sma, upper, lower, max_period):
    period = trial.suggest_int("period", 10, max_period)
    num_std = trial.suggest_float("num_std", 1.5, 3.0)
    sl = trial.suggest_float("sl_atr", 0.5, 4.0)
    tp = trial.suggest_float("tp_atr", 0.5, 4.0)
    cd = trial.suggest_int("cooldown", 1, 8)
    
    # Recompute BB with trial params
    sma_t, upper_t, lower_t = compute_bollinger(c, period, num_std)
    
    trades, _ = run_strategy(o, h, l, c, atr, sma_t, upper_t, lower_t, period, sl, tp, cd)
    if len(trades) < 5: return -9999
    wr = (trades > 0).mean()
    sh = trades.mean() / trades.std() * np.sqrt(252) if trades.std() > 0 else -9
    return sh * (1 + wr - 0.5) * np.sign(trades.sum() + 0.01)

# ============================================================
# OPTIMIZE + EXTRACT CHAMPIONS
# ============================================================
def optimize_tf(df, tf_name, max_period):
    split = df.index[-1] - pd.Timedelta(days=TEST_DAYS)
    train_end = split - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
    df_train = df[(df.index > train_start) & (df.index <= train_end)]
    if len(df_train) < 500:
        print(f"  {tf_name}: not enough data ({len(df_train)} bars)")
        return []
    
    o, h, l, c, atr = precompute(df_train)
    
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(lambda t: objective(t, o, h, l, c, atr, *compute_bollinger(c, 20, 2.0), max_period),
                   n_trials=N_TRIALS, show_progress_bar=False)
    
    trials = sorted([t for t in study.trials if t.value is not None and t.value > -9990],
                    key=lambda t: t.value, reverse=True)[:N_CHAMPIONS]
    
    champions = []
    for t in trials:
        p = t.params
        champions.append({
            "period": p["period"], "num_std": p["num_std"],
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
    print("Bollinger Bands Ensemble - Champion Optimization")
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
    print("\nOptimizing Bollinger Bands champions...")
    all_champions = {}
    for tf_name, (_, _, max_period) in TF_CONFIG.items():
        print(f"\n{tf_name}:")
        champs = optimize_tf(dfs[tf_name], tf_name, max_period)
        all_champions[tf_name] = champs
        for i, ch in enumerate(champs):
            print(f"  #{i+1}: period={ch['period']} std={ch['num_std']:.2f} "
                  f"sl={ch['sl_atr']:.2f} tp={ch['tp_atr']:.2f} cd={ch['cooldown']}")
    
    # Save champions
    with open("D:\\PYTHON\\METATRADER\\champions_bb.json", "w") as fp:
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
            sma_t, upper_t, lower_t = compute_bollinger(c, ch["period"], ch["num_std"])
            trades, eq = run_strategy(o, h, l, c, atr, sma_t, upper_t, lower_t,
                                       ch["period"], ch["sl_atr"], ch["tp_atr"], ch["cooldown"])
            p = sum(trades)
            tf_p += p; tf_t += len(trades); tf_w += (trades > 0).sum()
            
            eq_usd = eq * LOT * 10
            ax.plot(eq_usd, label=f"Ch{i+1} ({p:+.0f}p)", linewidth=0.8)
            
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
    ax.set_title(f"COMBINED BB: {grand_p:+.1f}p (${grand_p*LOT*10:+.2f}) | WR {gwr:.0f}%")
    ax.set_ylabel("PnL ($)")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Bollinger Bands Ensemble - Test Period (30 days)", fontsize=14)
    plt.tight_layout()
    plt.savefig("D:\\PYTHON\\METATRADER\\bb_test.png", dpi=150)
    
    print(f"\n{'='*70}")
    print(f"BB GRAND TOTAL: {grand_p:+.1f}p (${grand_p*LOT*10:+.2f}) | {grand_t} trades | WR {gwr:.0f}%")
    print(f"{'='*70}")
    print(f"\nSaved: bb_test.png + champions_bb.json")
