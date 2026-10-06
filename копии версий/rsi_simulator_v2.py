"""
RSI Live Simulator v2 - Fast bar-by-bar
Skip M5, use M15 bars directly. ~300 iterations instead of 6337.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import time

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
        return None, None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss < 1e-10:
        return 100.0, None
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Prev RSI
    gains_prev = np.where(deltas[:-1] > 0, deltas[:-1], 0)
    losses_prev = np.where(deltas[:-1] < 0, -deltas[:-1], 0)
    avg_gain_p = np.mean(gains_prev[-period:])
    avg_loss_p = np.mean(losses_prev[-period:])
    if avg_loss_p < 1e-10:
        return rsi, 100.0
    rs_p = avg_gain_p / avg_loss_p
    rsi_prev = 100 - (100 / (1 + rs_p))
    return rsi, rsi_prev

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 2:
        return None
    trs = np.maximum(highs[1:]-lows[1:], np.maximum(np.abs(highs[1:]-closes[:-1]), np.abs(lows[1:]-closes[:-1])))
    return np.mean(trs[-period:])

class Position:
    def __init__(self, magic, direction, entry, sl, tp, bar_time):
        self.magic = magic
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.open_bar = bar_time

class Simulator:
    def __init__(self):
        self.positions = {}
        self.trades = []
        self.equity = [0.0]
        self.last_sl_bar = {}
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
    
    def close_trade(self, magic, price, bar_time, reason=""):
        if magic not in self.positions:
            return 0
        pos = self.positions.pop(magic)
        if pos.direction == 1:
            pnl = (price - pos.entry) / PIPIP - COST
        else:
            pnl = (pos.entry - price) / PIPIP - COST
        self.trades.append((pnl, magic, bar_time, reason))
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
            pnl = self.close_trade(magic, price, bar_time, reason)
            if reason == "SL":
                self.last_sl_bar[magic] = bar_time

def run():
    print("="*60)
    print("RSI LIVE SIMULATOR v2 - Fast Mode")
    print("="*60)
    
    # Load all TFs
    print("\nLoading data...")
    dfs = {}
    for tf_name, tf_data in CHAMPIONS.items():
        tf_val = tf_data["tf"]
        n = 30000 if tf_name == "M15" else 20000
        raw = load_data(tf_val, n)
        if raw is not None:
            dfs[tf_name] = raw.dropna()
            print(f"  {tf_name}: {len(dfs[tf_name])} bars")
    
    test_days = 30
    sim = Simulator()
    
    # Use M15 as primary timeline (fastest)
    df_m15 = dfs["M15"]
    split = df_m15.index[-1] - pd.Timedelta(days=test_days)
    timeline = df_m15[df_m15.index >= split].index
    
    print(f"\nTimeline: {len(timeline)} M15 bars")
    print(f"Period: {timeline[0]} to {timeline[-1]}")
    print(f"\nSimulating...")
    
    start_time = time.time()
    
    for bar_idx, bar_time in enumerate(timeline):
        if bar_idx % 50 == 0:
            elapsed = time.time() - start_time
            pct = bar_idx / len(timeline) * 100
            print(f"  {pct:.0f}% | Bar {bar_idx}/{len(timeline)} | Pos: {len(sim.positions)} | Bal: ${sim.balance:.2f}")
        
        for tf_name, tf_data in CHAMPIONS.items():
            df = dfs[tf_name]
            available = df[df.index <= bar_time]
            if len(available) < 30:
                continue
            
            closes = available["close"].values
            highs = available["high"].values
            lows = available["low"].values
            
            cur_high = highs[-1]
            cur_low = lows[-1]
            cur_close = closes[-1]
            
            # Check SL/TP
            sim.check_sl_tp(cur_high, cur_low, bar_time)
            
            # Indicators
            rsi, rsi_prev = calc_rsi(closes, tf_data["rsi_period"])
            atr = calc_atr(highs, lows, closes)
            
            if rsi is None or rsi_prev is None or atr is None:
                continue
            
            # Champions
            for ch in tf_data["champs"]:
                magic = sim.get_magic(tf_name, ch["id"])
                
                if magic not in sim.positions:
                    # Buy
                    if rsi > ch["buy_thr"] and rsi_prev <= ch["buy_thr"]:
                        sim.open_trade(magic, 1, cur_close, atr, ch["sl"], ch["tp"], bar_time)
                    # Sell
                    elif rsi < ch["sell_thr"] and rsi_prev >= ch["sell_thr"]:
                        sim.open_trade(magic, -1, cur_close, atr, ch["sl"], ch["tp"], bar_time)
                else:
                    pos = sim.positions[magic]
                    if pos.direction == 1 and (rsi > 70 or rsi < 50):
                        sim.close_trade(magic, cur_close, bar_time, "EXIT")
                    elif pos.direction == -1 and (rsi < 30 or rsi > 50):
                        sim.close_trade(magic, cur_close, bar_time, "EXIT")
    
    # Close all remaining
    last_close = dfs["M15"]["close"].values[-1]
    for magic in list(sim.positions.keys()):
        sim.close_trade(magic, last_close, timeline[-1], "EOD")
    
    elapsed = time.time() - start_time
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Time: {elapsed:.1f}s")
    print(f"Trades: {len(sim.trades)}")
    print(f"Balance: ${sim.balance:.2f}")
    
    if sim.trades:
        pnls = [t[0] for t in sim.trades]
        wins = sum(1 for p in pnls if p > 0)
        print(f"WR: {wins/len(pnls)*100:.1f}%")
        print(f"Total PnL: {sum(pnls):+.1f}p")
        
        # By TF
        print(f"\n{'TF':<6} {'Trades':>8} {'PnL':>10} {'WR':>6}")
        print("-"*35)
        for tf_idx, tf_name in enumerate(CHAMPIONS.keys()):
            tf_trades = [t for t in sim.trades if MAGIC_BASE + tf_idx * 10 <= t[1] < MAGIC_BASE + (tf_idx+1) * 10]
            if tf_trades:
                tf_pnl = sum(t[0] for t in tf_trades)
                tf_wr = sum(1 for t in tf_trades if t[0] > 0) / len(tf_trades) * 100
                print(f"{tf_name:<6} {len(tf_trades):>8} {tf_pnl:>+10.1f} {tf_wr:>5.0f}%")
        
        # By reason
        print(f"\n{'Reason':<10} {'Count':>8} {'PnL':>10}")
        print("-"*30)
        for reason in ["SL", "TP", "EXIT", "EOD"]:
            r_trades = [t for t in sim.trades if t[3] == reason]
            if r_trades:
                r_pnl = sum(t[0] for t in r_trades)
                print(f"{reason:<10} {len(r_trades):>8} {r_pnl:>+10.1f}")
    
    # Save
    results = {"trades": len(sim.trades), "balance": sim.balance, "pnls": [t[0] for t in sim.trades]}
    with open("D:\\PYTHON\\METATRADER\\rsi_sim_results.json", "w") as f:
        json.dump(results, f)
    
    # Plot
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
    except Exception as e:
        print(f"Plot error: {e}")

if __name__ == "__main__":
    run()
