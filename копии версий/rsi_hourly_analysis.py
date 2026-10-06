"""
RSI Hourly Analysis - Find optimal trading hours
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
# RSI STRATEGY
# ============================================================
def run_rsi_hourly(o, h, l, c, atr, rsi, times, buy_thr, sell_thr, sl_atr, tp_atr, cooldown=3):
    n = len(c)
    pos = 0; ep = sl_v = tp_v = 0.0; ei = 0
    trades = []; trade_hours = []
    last_sl_bar = -999
    s = 20
    
    for i in range(s, n):
        if pos == 0:
            if np.isnan(rsi[i]): continue
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
                trade_hours.append(times[ei].hour)
    return np.array(trades), np.array(trade_hours)

# ============================================================
# LOAD CHAMPIONS
# ============================================================
def load_champions():
    with open("D:\\PYTHON\\METATRADER\\champions_rsi.json") as f:
        return json.load(f)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try: mt5.shutdown()
    except: pass
    
    print("="*70)
    print("RSI Hourly Analysis - Find Optimal Trading Hours")
    print(f"Test period: last {TEST_DAYS} days")
    print("="*70)
    
    # Load data (H1 for hourly analysis)
    print("\nLoading H1 data...")
    df = load_data(mt5.TIMEFRAME_H1, 20000)
    df = df.dropna()
    print(f"  H1: {len(df)} bars")
    
    # Test period
    split = df.index[-1] - pd.Timedelta(days=TEST_DAYS)
    test = df[df.index >= split]
    o, h, l, c, atr = precompute(test)
    rsi = compute_rsi(c, 14)
    times = test.index
    
    # Load champions
    champions = load_champions()
    
    # ============================================================
    # HOURLY PnL ANALYSIS
    # ============================================================
    print("\nRunning hourly analysis...")
    
    hourly_pnl = {}
    hourly_trades = {}
    for hour in range(24):
        hourly_pnl[hour] = 0.0
        hourly_trades[hour] = 0
    
    # Run all champions
    for ch in champions.get("H1", []):
        trades, trade_hours = run_rsi_hourly(o, h, l, c, atr, rsi, times,
                                              ch["buy_thr"], ch["sell_thr"],
                                              ch["sl_atr"], ch["tp_atr"], ch["cooldown"])
        for pnl, hour in zip(trades, trade_hours):
            hourly_pnl[hour] += pnl
            hourly_trades[hour] += 1
    
    # Also run M15 and M30 for full picture
    for tf_name, tf_val in [("M15", mt5.TIMEFRAME_M15), ("M30", mt5.TIMEFRAME_M30)]:
        print(f"\nLoading {tf_name} data...")
        df_tf = load_data(tf_val, 30000 if tf_name == "M15" else 20000)
        df_tf = df_tf.dropna()
        split_tf = df_tf.index[-1] - pd.Timedelta(days=TEST_DAYS)
        test_tf = df_tf[df_tf.index >= split_tf]
        o_tf, h_tf, l_tf, c_tf, atr_tf = precompute(test_tf)
        rsi_tf = compute_rsi(c_tf, 14)
        times_tf = test_tf.index
        
        for ch in champions.get(tf_name, []):
            trades, trade_hours = run_rsi_hourly(o_tf, h_tf, l_tf, c_tf, atr_tf, rsi_tf, times_tf,
                                                  ch["buy_thr"], ch["sell_thr"],
                                                  ch["sl_atr"], ch["tp_atr"], ch["cooldown"])
            for pnl, hour in zip(trades, trade_hours):
                hourly_pnl[hour] += pnl
                hourly_trades[hour] += 1
    
    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "="*70)
    print("HOURLY PnL ANALYSIS (all TFs combined)")
    print("="*70)
    print(f"{'Hour':>6} {'PnL':>10} {'Trades':>8} {'Avg/Trade':>10}")
    print("-"*40)
    
    for hour in range(24):
        pnl = hourly_pnl[hour]
        trades = hourly_trades[hour]
        avg = pnl / trades if trades > 0 else 0
        marker = " <--" if pnl > 50 else (" <--" if pnl < -50 else "")
        print(f"{hour:6d} {pnl:>+10.1f} {trades:>8d} {avg:>+10.2f}{marker}")
    
    total_pnl = sum(hourly_pnl.values())
    total_trades = sum(hourly_trades.values())
    print("-"*40)
    print(f"{'TOTAL':>6} {total_pnl:>+10.1f} {total_trades:>8d}")
    
    # ============================================================
    # FIND OPTIMAL WINDOWS
    # ============================================================
    print("\n" + "="*70)
    print("OPTIMAL SESSION WINDOWS")
    print("="*70)
    
    # Test different session windows
    windows = [
        (0, 24, "24h"),
        (0, 8, "00-08 (night)"),
        (8, 12, "08-12 (morning)"),
        (12, 17, "12-17 (afternoon)"),
        (17, 24, "17-24 (evening)"),
        (8, 17, "08-17 (day)"),
        (8, 16, "08-16 (MA session)"),
        (9, 18, "09-18"),
        (10, 20, "10-20"),
        (2, 10, "02-10 (Asian+EU)"),
        (7, 15, "07-15 (EU)"),
        (13, 21, "13-21 (EU+US)"),
    ]
    
    print(f"\n{'Session':<20} {'PnL':>10} {'Trades':>8} {'$/Trade':>10}")
    print("-"*55)
    
    best_pnl = -9999
    best_window = ""
    
    for start, end, name in windows:
        window_pnl = 0
        window_trades = 0
        for hour in range(24):
            if start <= end:
                if start <= hour < end:
                    window_pnl += hourly_pnl[hour]
                    window_trades += hourly_trades[hour]
            else:
                if hour >= start or hour < end:
                    window_pnl += hourly_pnl[hour]
                    window_trades += hourly_trades[hour]
        
        avg = window_pnl / window_trades if window_trades > 0 else 0
        marker = " <-- BEST" if window_pnl > best_pnl else ""
        print(f"{name:<20} {window_pnl:>+10.1f} {window_trades:>8d} {avg:>+10.2f}{marker}")
        
        if window_pnl > best_pnl:
            best_pnl = window_pnl
            best_window = name
    
    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Hourly PnL bars
    hours = list(range(24))
    pnls = [hourly_pnl[h] for h in hours]
    colors = ["green" if p > 0 else "red" for p in pnls]
    axes[0].bar(hours, pnls, color=colors, alpha=0.7)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Hour (server time)")
    axes[0].set_ylabel("PnL (pips)")
    axes[0].set_title("RSI - Hourly PnL Distribution")
    axes[0].set_xticks(hours)
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative PnL
    cum_pnl = np.cumsum(pnls)
    axes[1].plot(hours, cum_pnl, "b-", linewidth=2)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].fill_between(hours, cum_pnl, alpha=0.3)
    axes[1].set_xlabel("Hour (server time)")
    axes[1].set_ylabel("Cumulative PnL (pips)")
    axes[1].set_title("RSI - Cumulative PnL by Hour")
    axes[1].set_xticks(hours)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("D:\\PYTHON\\METATRADER\\rsi_hourly.png", dpi=150)
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION: Trade {best_window}")
    print(f"{'='*70}")
    print(f"\nSaved: rsi_hourly.png")
