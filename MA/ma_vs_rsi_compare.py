"""
MA vs RSI vs MA+RSI Ensemble - Comparison
All three on same 30-day test period
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SYMBOL = "EURUSD"
PIPIP = 0.0001
COST = 0.7
LOT = 0.01
TF_CONFIG = {
    "M15": (mt5.TIMEFRAME_M15, 30000),
    "M30": (mt5.TIMEFRAME_M30, 20000),
    "H1":  (mt5.TIMEFRAME_H1,  20000),
}
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

def compute_rsi(c, period):
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean().values
    avg_loss = pd.Series(loss).rolling(period).mean().values
    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================================================
# STRATEGIES
# ============================================================
def run_ma(o, h, l, c, atr, fast, slow, entry_thr, sl_atr, tp_atr, cooldown=3):
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
                if l[i] <= sl_v: pnl = -(sl_v - ep)/PIPIP - COST; pos = 0; last_sl_bar = i
                elif h[i] >= tp_v: pnl = (tp_v - ep)/PIPIP - COST; pos = 0
                elif d > -entry_thr*0.3 or (b >= 3 and d <= 0): pnl = (c[i] - ep)/PIPIP - COST; pos = 0
            else:
                if h[i] >= sl_v: pnl = (ep - sl_v)/PIPIP - COST; pos = 0; last_sl_bar = i
                elif l[i] <= tp_v: pnl = (ep - tp_v)/PIPIP - COST; pos = 0
                elif d < entry_thr*0.3 or (b >= 3 and d >= 0): pnl = (ep - c[i])/PIPIP - COST; pos = 0
            if pnl != 0:
                trades.append(pnl)
                bal += pnl * LOT * 10
        equities.append(bal)
    return np.array(trades), np.array(equities)

def run_rsi(o, h, l, c, atr, rsi, buy_thr, sell_thr, sl_atr, tp_atr, cooldown=3):
    n = len(c)
    pos = 0; ep = sl_v = tp_v = 0.0; ei = 0
    trades = []; equities = [0.0]; bal = 0.0
    last_sl_bar = -999
    s = 20
    
    for i in range(s, n):
        if pos == 0:
            if np.isnan(rsi[i]): 
                equities.append(bal)
                continue
            cooled = (last_sl_bar >= 0 and (i - last_sl_bar) < cooldown)
            if rsi[i] > buy_thr and rsi[i-1] <= buy_thr and (i+1) < n and not cooled:
                pos = 1; ep = o[i+1]; sl_v = ep - atr[i]*sl_atr; tp_v = ep + atr[i]*tp_atr; ei = i+1
            elif rsi[i] < sell_thr and rsi[i-1] >= sell_thr and (i+1) < n and not cooled:
                pos = -1; ep = o[i+1]; sl_v = ep + atr[i]*sl_atr; tp_v = ep - atr[i]*tp_atr; ei = i+1
        else:
            b = i - ei
            pnl = 0
            if pos == 1:
                if l[i] <= sl_v: pnl = -(sl_v - ep)/PIPIP - COST; pos = 0; last_sl_bar = i
                elif h[i] >= tp_v: pnl = (tp_v - ep)/PIPIP - COST; pos = 0
                elif rsi[i] > 70 or (b >= 3 and rsi[i] < 50): pnl = (c[i] - ep)/PIPIP - COST; pos = 0
            else:
                if h[i] >= sl_v: pnl = (ep - sl_v)/PIPIP - COST; pos = 0; last_sl_bar = i
                elif l[i] <= tp_v: pnl = (ep - tp_v)/PIPIP - COST; pos = 0
                elif rsi[i] < 30 or (b >= 3 and rsi[i] > 50): pnl = (ep - c[i])/PIPIP - COST; pos = 0
            if pnl != 0:
                trades.append(pnl)
                bal += pnl * LOT * 10
        equities.append(bal)
    return np.array(trades), np.array(equities)

# ============================================================
# LOAD CHAMPIONS
# ============================================================
def load_champions():
    with open("D:\\PYTHON\\METATRADER\\champions_corrected.json") as f:
        ma_champs = json.load(f)
    
    with open("D:\\PYTHON\\METATRADER\\champions_rsi.json") as f:
        rsi_champs = json.load(f)
    
    return ma_champs, rsi_champs

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try: mt5.shutdown()
    except: pass
    
    print("="*70)
    print("COMPARISON: MA vs RSI vs MA+RSI Ensemble")
    print(f"Test period: last {TEST_DAYS} days")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    dfs = {}
    for tf_name, (tf_val, n_bars) in TF_CONFIG.items():
        raw = load_data(tf_val, n_bars)
        if raw is not None:
            dfs[tf_name] = raw.dropna()
            print(f"  {tf_name}: {len(dfs[tf_name])} bars")
    
    # Load champions
    ma_champs, rsi_champs = load_champions()
    print(f"\nMA champions: {sum(len(v) for v in ma_champs.values())} total")
    print(f"RSI champions: {sum(len(v) for v in rsi_champs.values())} total")
    
    # ============================================================
    # RUN ALL THREE STRATEGIES
    # ============================================================
    results = {
        "MA": {"pips": 0, "trades": 0, "wins": 0, "equity": []},
        "RSI": {"pips": 0, "trades": 0, "wins": 0, "equity": []},
        "MA+RSI": {"pips": 0, "trades": 0, "wins": 0, "equity": []},
    }
    
    for tf_name in TF_CONFIG.keys():
        df = dfs[tf_name]
        split = df.index[-1] - pd.Timedelta(days=TEST_DAYS)
        test = df[df.index >= split]
        o, h, l, c, atr = precompute(test)
        rsi = compute_rsi(c, 14)
        
        print(f"\n{tf_name}:")
        
        # MA champions
        for ch in ma_champs.get(tf_name, []):
            trades, eq = run_ma(o, h, l, c, atr, ch["fast"], ch["slow"],
                                ch["entry"], ch["sl_atr"], ch["tp_atr"], ch["cooldown"])
            p = sum(trades)
            results["MA"]["pips"] += p
            results["MA"]["trades"] += len(trades)
            results["MA"]["wins"] += (trades > 0).sum()
            eq_usd = eq * LOT * 10
            if len(results["MA"]["equity"]) == 0:
                results["MA"]["equity"] = eq_usd.copy()
            else:
                new_len = max(len(results["MA"]["equity"]), len(eq_usd))
                new_eq = np.zeros(new_len)
                new_eq[:len(results["MA"]["equity"])] += results["MA"]["equity"]
                new_eq[:len(eq_usd)] += eq_usd
                results["MA"]["equity"] = new_eq
        
        # RSI champions
        for ch in rsi_champs.get(tf_name, []):
            trades, eq = run_rsi(o, h, l, c, atr, rsi, ch["buy_thr"], ch["sell_thr"],
                                  ch["sl_atr"], ch["tp_atr"], ch["cooldown"])
            p = sum(trades)
            results["RSI"]["pips"] += p
            results["RSI"]["trades"] += len(trades)
            results["RSI"]["wins"] += (trades > 0).sum()
            eq_usd = eq * LOT * 10
            if len(results["RSI"]["equity"]) == 0:
                results["RSI"]["equity"] = eq_usd.copy()
            else:
                new_len = max(len(results["RSI"]["equity"]), len(eq_usd))
                new_eq = np.zeros(new_len)
                new_eq[:len(results["RSI"]["equity"])] += results["RSI"]["equity"]
                new_eq[:len(eq_usd)] += eq_usd
                results["RSI"]["equity"] = new_eq
        
        # MA+RSI combined (each TF has both MA and RSI champions)
        for ch in ma_champs.get(tf_name, []):
            trades, eq = run_ma(o, h, l, c, atr, ch["fast"], ch["slow"],
                                ch["entry"], ch["sl_atr"], ch["tp_atr"], ch["cooldown"])
            p = sum(trades)
            results["MA+RSI"]["pips"] += p
            results["MA+RSI"]["trades"] += len(trades)
            results["MA+RSI"]["wins"] += (trades > 0).sum()
            eq_usd = eq * LOT * 10
            if len(results["MA+RSI"]["equity"]) == 0:
                results["MA+RSI"]["equity"] = eq_usd.copy()
            else:
                new_len = max(len(results["MA+RSI"]["equity"]), len(eq_usd))
                new_eq = np.zeros(new_len)
                new_eq[:len(results["MA+RSI"]["equity"])] += results["MA+RSI"]["equity"]
                new_eq[:len(eq_usd)] += eq_usd
                results["MA+RSI"]["equity"] = new_eq
        
        for ch in rsi_champs.get(tf_name, []):
            trades, eq = run_rsi(o, h, l, c, atr, rsi, ch["buy_thr"], ch["sell_thr"],
                                  ch["sl_atr"], ch["tp_atr"], ch["cooldown"])
            p = sum(trades)
            results["MA+RSI"]["pips"] += p
            results["MA+RSI"]["trades"] += len(trades)
            results["MA+RSI"]["wins"] += (trades > 0).sum()
            eq_usd = eq * LOT * 10
            if len(results["MA+RSI"]["equity"]) == 0:
                results["MA+RSI"]["equity"] = eq_usd.copy()
            else:
                new_len = max(len(results["MA+RSI"]["equity"]), len(eq_usd))
                new_eq = np.zeros(new_len)
                new_eq[:len(results["MA+RSI"]["equity"])] += results["MA+RSI"]["equity"]
                new_eq[:len(eq_usd)] += eq_usd
                results["MA+RSI"]["equity"] = new_eq
    
    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "="*70)
    print("FINAL RESULTS (30 days)")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["blue", "red", "green"]
    
    for idx, (name, color) in enumerate(zip(["MA", "RSI", "MA+RSI"], colors)):
        r = results[name]
        wr = r["wins"]/r["trades"]*100 if r["trades"] > 0 else 0
        usd = r["pips"] * LOT * 10
        print(f"\n{name}:")
        print(f"  Pips: {r['pips']:+.1f}")
        print(f"  USD:  ${usd:+.2f}")
        print(f"  Trades: {r['trades']}")
        print(f"  WR: {wr:.0f}%")
        
        ax = axes[idx]
        ax.plot(r["equity"], color=color, linewidth=1.2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{name}: {r['pips']:+.1f}p (${usd:+.2f})\n{r['trades']} trades | WR {wr:.0f}%")
        ax.set_ylabel("PnL ($)")
        ax.set_xlabel("Bars")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("MA vs RSI vs MA+RSI - 30 Day Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("D:\\PYTHON\\METATRADER\\ma_vs_rsi.png", dpi=150)
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Strategy':<12} {'Pips':>8} {'USD':>10} {'Trades':>8} {'WR':>6}")
    print(f"{'-'*70}")
    for name in ["MA", "RSI", "MA+RSI"]:
        r = results[name]
        wr = r["wins"]/r["trades"]*100 if r["trades"] > 0 else 0
        usd = r["pips"] * LOT * 10
        print(f"{name:<12} {r['pips']:>+8.1f} ${usd:>+9.2f} {r['trades']:>8} {wr:>5.0f}%")
    print(f"{'='*70}")
    print(f"\nSaved: ma_vs_rsi.png")
