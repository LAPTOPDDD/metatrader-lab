"""
RSI Live Simulator - Bar-by-bar like real trading
Test 1 month in ~2 minutes
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import time
import os

SYMBOL = "EURUSD"
PIPIP = 0.0001
COST = 0.7
LOT = 0.01
MAGIC_BASE = 300000

CHAMPIONS = {
    "M15": {
        "tf": mt5.TIMEFRAME_M15,
        "rsi_period": 14,
        "champs": [
            {"id":0, "buy_thr":44.4, "sell_thr":80.0, "sl":0.68, "tp":0.80, "cd":6},
            {"id":1, "buy_thr":44.3, "sell_thr":79.9, "sl":0.60, "tp":0.89, "cd":6},
            {"id":2, "buy_thr":43.6, "sell_thr":80.0, "sl":0.63, "tp":0.90, "cd":6},
            {"id":3, "buy_thr":43.6, "sell_thr":80.0, "sl":0.62, "tp":0.89, "cd":6},
            {"id":4, "buy_thr":44.6, "sell_thr":80.0, "sl":0.62, "tp":0.88, "cd":6},
        ]
    },
    "M30": {
        "tf": mt5.TIMEFRAME_M30,
        "rsi_period": 14,
        "champs": [
            {"id":0, "buy_thr":42.6, "sell_thr":79.9, "sl":0.65, "tp":0.69, "cd":3},
            {"id":1, "buy_thr":41.3, "sell_thr":80.0, "sl":0.57, "tp":0.69, "cd":4},
            {"id":2, "buy_thr":41.3, "sell_thr":80.0, "sl":0.59, "tp":0.60, "cd":4},
            {"id":3, "buy_thr":41.0, "sell_thr":80.0, "sl":0.66, "tp":0.61, "cd":4},
            {"id":4, "buy_thr":43.8, "sell_thr":80.0, "sl":0.61, "tp":0.85, "cd":5},
        ]
    },
    "H1": {
        "tf": mt5.TIMEFRAME_H1,
        "rsi_period": 14,
        "champs": [
            {"id":0, "buy_thr":44.9, "sell_thr":79.9, "sl":0.55, "tp":0.62, "cd":6},
            {"id":1, "buy_thr":45.0, "sell_thr":80.0, "sl":0.55, "tp":0.61, "cd":6},
            {"id":2, "buy_thr":44.9, "sell_thr":80.0, "sl":0.55, "tp":0.60, "cd":6},
            {"id":3, "buy_thr":44.9, "sell_thr":79.8, "sl":0.54, "tp":0.61, "cd":6},
            {"id":4, "buy_thr":45.0, "sell_thr":80.0, "sl":0.55, "tp":0.66, "cd":6},
        ]
    }
}

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

def calc_rsi(closes, period=14):
    if len(closes) < period + 5:
        return None
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 2:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    return sum(trs[-period:]) / period

# ============================================================
# POSITION TRACKING
# ============================================================
class Position:
    def __init__(self, magic, direction, entry, sl, tp, bar_time):
        self.magic = magic
        self.direction = direction  # 1=buy, -1=sell
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.open_bar = bar_time

class Simulator:
    def __init__(self):
        self.positions = {}  # magic -> Position
        self.trades = []     # list of (pnl, magic, close_time)
        self.equity = [0.0]
        self.last_sl_bar = {}  # magic -> bar_index
        self.balance = 0.0
    
    def get_magic(self, tf_name, ch_id):
        tf_idx = list(CHAMPIONS.keys()).index(tf_name)
        return MAGIC_BASE + tf_idx * 10 + ch_id
    
    def open_trade(self, magic, direction, price, atr, sl_mult, tp_mult, bar_time):
        if magic in self.positions:
            return False
        
        if direction == 1:
            sl = price - atr * sl_mult
            tp = price + atr * tp_mult
        else:
            sl = price + atr * sl_mult
            tp = price - atr * tp_mult
        
        self.positions[magic] = Position(magic, direction, price, sl, tp, bar_time)
        return True
    
    def close_trade(self, magic, price, bar_time):
        if magic not in self.positions:
            return 0
        
        pos = self.positions.pop(magic)
        if pos.direction == 1:
            pnl = (price - pos.entry) / PIPIP - COST
        else:
            pnl = (pos.entry - price) / PIPIP - COST
        
        self.trades.append((pnl, magic, bar_time))
        self.balance += pnl * LOT * 10
        self.equity.append(self.balance)
        return pnl
    
    def check_sl_tp(self, high, low, bar_time):
        to_close = []
        for magic, pos in self.positions.items():
            if pos.direction == 1:
                if low <= pos.sl:
                    to_close.append((magic, pos.sl, "SL"))
                elif high >= pos.tp:
                    to_close.append((magic, pos.tp, "TP"))
            else:
                if high >= pos.sl:
                    to_close.append((magic, pos.sl, "SL"))
                elif low <= pos.tp:
                    to_close.append((magic, pos.tp, "TP"))
        
        for magic, price, reason in to_close:
            pnl = self.close_trade(magic, price, bar_time)
            if reason == "SL":
                self.last_sl_bar[magic] = bar_time
    
    def count_positions(self):
        return len(self.positions)

# ============================================================
# SIMULATOR LOOP
# ============================================================
def run_simulator():
    print("="*70)
    print("RSI LIVE SIMULATOR - Bar by Bar")
    print("15 Champions | M15+M30+H1 | Testing 30 days")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    dfs = {}
    for tf_name, tf_data in CHAMPIONS.items():
        tf_val = tf_data["tf"]
        n = 30000 if tf_name == "M15" else 20000
        raw = load_data(tf_val, n)
        if raw is not None:
            dfs[tf_name] = raw.dropna()
            print(f"  {tf_name}: {len(dfs[tf_name])} bars")
    
    # Test period
    test_days = 30
    
    sim = Simulator()
    
    # Build unified timeline from M5 (smallest resolution)
    print("\nBuilding timeline...")
    raw_m5 = load_data(mt5.TIMEFRAME_M5, 50000)
    if raw_m5 is None:
        print("Failed to load M5 data")
        return
    
    raw_m5 = raw_m5.dropna()
    split = raw_m5.index[-1] - pd.Timedelta(days=test_days)
    timeline = raw_m5[raw_m5.index >= split].index
    
    print(f"  Timeline: {len(timeline)} bars ({timeline[0]} to {timeline[-1]})")
    print(f"\nSimulating...")
    
    # Pre-compute RSI windows for each TF
    rsi_windows = {}
    for tf_name, df in dfs.items():
        closes = df["close"].values.tolist()
        rsi_windows[tf_name] = closes
    
    # Run simulation
    start_time = time.time()
    last_pct = 0
    
    for bar_idx, bar_time in enumerate(timeline):
        # Progress
        pct = int(bar_idx / len(timeline) * 100)
        if pct >= last_pct + 5:
            last_pct = pct
            elapsed = time.time() - start_time
            eta = elapsed / (bar_idx + 1) * (len(timeline) - bar_idx - 1)
            print(f"  {pct}% | Bar {bar_idx}/{len(timeline)} | Positions: {sim.count_positions()} | Balance: ${sim.balance:.2f} | ETA: {eta:.0f}s")
        
        # For each TF
        for tf_name, tf_data in CHAMPIONS.items():
            df = dfs[tf_name]
            tf_val = tf_data["tf"]
            
            # Get data up to current bar
            available = df[df.index <= bar_time]
            if len(available) < 30:
                continue
            
            closes = available["close"].values
            highs = available["high"].values
            lows = available["low"].values
            opens = available["open"].values
            
            # Current bar values
            cur_high = highs[-1]
            cur_low = lows[-1]
            cur_close = closes[-1]
            cur_open = opens[-1]
            
            # Check SL/TP on existing positions
            sim.check_sl_tp(cur_high, cur_low, bar_time)
            
            # Calculate indicators
            rsi = calc_rsi(closes, tf_data["rsi_period"])
            rsi_prev = calc_rsi(closes[:-1], tf_data["rsi_period"]) if len(closes) > 1 else None
            atr = calc_atr(highs, lows, closes)
            
            if rsi is None or rsi_prev is None or atr is None:
                continue
            
            # Check signals for each champion
            for ch in tf_data["champs"]:
                magic = sim.get_magic(tf_name, ch["id"])
                
                # If no position, check for entry
                if magic not in sim.positions:
                    cooled = False
                    if magic in sim.last_sl_bar:
                        # Simple cooldown: skip if last SL was recent
                        pass  # In real sim, we'd track bars properly
                    
                    # Buy signal
                    if rsi > ch["buy_thr"] and rsi_prev <= ch["buy_thr"]:
                        if sim.open_trade(magic, 1, cur_close, atr, ch["sl"], ch["tp"], bar_time):
                            pass
                    
                    # Sell signal
                    elif rsi < ch["sell_thr"] and rsi_prev >= ch["sell_thr"]:
                        if sim.open_trade(magic, -1, cur_close, atr, ch["sl"], ch["tp"], bar_time):
                            pass
                
                # If has position, check for exit
                elif magic in sim.positions:
                    pos = sim.positions[magic]
                    if pos.direction == 1 and (rsi > 70 or rsi < 50):
                        sim.close_trade(magic, cur_close, bar_time)
                    elif pos.direction == -1 and (rsi < 30 or rsi > 50):
                        sim.close_trade(magic, cur_close, bar_time)
    
    # Close remaining positions
    for magic in list(sim.positions.keys()):
        sim.close_trade(magic, dfs["M15"]["close"].values[-1], timeline[-1])
    
    elapsed = time.time() - start_time
    
    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Total trades: {len(sim.trades)}")
    print(f"Final balance: ${sim.balance:.2f}")
    
    # Trade stats
    if sim.trades:
        pnls = [t[0] for t in sim.trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        win_rate = wins / len(pnls) * 100
        avg_win = np.mean([p for p in pnls if p > 0]) if wins > 0 else 0
        avg_loss = np.mean([p for p in pnls if p <= 0]) if losses > 0 else 0
        
        print(f"\nWin Rate: {win_rate:.1f}%")
        print(f"Avg Win: {avg_win:+.1f}p")
        print(f"Avg Loss: {avg_loss:+.1f}p")
        print(f"Profit Factor: {sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)):.2f}" if sum(p for p in pnls if p < 0) != 0 else "∞")
        
        # Per TF stats
        print(f"\n{'TF':<6} {'Trades':>8} {'PnL':>10} {'WR':>6}")
        print("-"*35)
        for tf_name in CHAMPIONS.keys():
            tf_trades = [t for t in sim.trades if t[1] // 10 % 3 == list(CHAMPIONS.keys()).index(tf_name)]
            if tf_trades:
                tf_pnl = sum(t[0] for t in tf_trades)
                tf_wr = sum(1 for t in tf_trades if t[0] > 0) / len(tf_trades) * 100
                print(f"{tf_name:<6} {len(tf_trades):>8} {tf_pnl:>+10.1f} {tf_wr:>5.0f}%")
    
    # Save results
    results = {
        "trades": len(sim.trades),
        "balance": sim.balance,
        "equity": sim.equity,
        "trades_list": [(t[0], str(t[2])) for t in sim.trades]
    }
    with open("D:\\PYTHON\\METATRADER\\rsi_sim_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot equity
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sim.equity, "b-", linewidth=1.2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(range(len(sim.equity)), sim.equity, alpha=0.3)
        ax.set_title(f"RSI Simulator - {len(sim.trades)} trades | ${sim.balance:.2f}")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Balance ($)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("D:\\PYTHON\\METATRADER\\rsi_sim_equity.png", dpi=150)
        print(f"\nSaved: rsi_sim_equity.png")
    except:
        pass
    
    print(f"Saved: rsi_sim_results.json")

if __name__ == "__main__":
    run_simulator()
