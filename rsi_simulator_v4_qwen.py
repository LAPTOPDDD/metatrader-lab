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

def init_mt5():
    if not mt5.initialize():
        raise Exception(f"MT5 initialize failed, error: {mt5.last_error()}")
    mt5.symbol_select(SYMBOL, True)

def load_data(tf, n_bars):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df[df.index.dayofweek < 5] # Убираем выходные
    return df[["open", "high", "low", "close"]].copy()

def prepare_indicators(df, period=14):
    """Векторизованный расчет RSI и ATR (Wilder's) один раз для всего DF"""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    df["avg_gain"] = gain.ewm(alpha=1/period, min_periods=period).mean()
    df["avg_loss"] = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    # Защита от деления на ноль
    df["rsi"] = 100 - (100 / (1 + df["avg_gain"] / df["avg_loss"].replace(0, 1e-10)))
    df["rsi_prev"] = df["rsi"].shift(1)
    
    tr1 = df["high"] - df["low"]
    tr2 = np.abs(df["high"] - df["close"].shift(1))
    tr3 = np.abs(df["low"] - df["close"].shift(1))
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr"] = df["tr"].ewm(alpha=1/period, min_periods=period).mean()
    
    return df

class Simulator:
    def __init__(self):
        self.positions = {}
        self.trades = []
        self.equity = [0.0]
        self.balance = 0.0
        self.pending = {}
        self.last_sl_bar = {}  # magic -> bar_time (datetime)
    
    def get_magic(self, tf_name, ch_id):
        tf_idx = list(CHAMPIONS.keys()).index(tf_name)
        return MAGIC_BASE + tf_idx * 10 + ch_id
    
    def process_pending(self, open_price, bar_time, tf_name):
        for magic, (direction, atr, sl_mult, tp_mult, pending_tf) in list(self.pending.items()):
            if magic in self.positions:
                del self.pending[magic]
                continue
            
            if direction == 1:
                sl = open_price - atr * sl_mult
                tp = open_price + atr * tp_mult
            else:
                sl = open_price + atr * sl_mult
                tp = open_price - atr * tp_mult
            
            self.positions[magic] = {
                "dir": direction, "entry": open_price,
                "sl": sl, "tp": tp, "bar": bar_time,
                "tf_name": pending_tf
            }
            del self.pending[magic]
    
    def close_trade(self, magic, price, bar_time, reason=""):
        if magic not in self.positions:
            return 0
        pos = self.positions.pop(magic)
        if pos["dir"] == 1:
            pnl = (price - pos["entry"]) / PIPIP - COST
        else:
            pnl = (pos["entry"] - price) / PIPIP - COST
            
        self.trades.append((pnl, magic, bar_time, reason))
        self.balance += pnl * LOT * 10
        self.equity.append(self.balance)
        return pnl
    
    def check_sl_tp(self, cur_open, cur_high, cur_low, bar_time):
        to_close = []
        for magic, pos in self.positions.items():
            if pos["dir"] == 1:
                if cur_low <= pos["sl"]:
                    # Консервативный учет гэпа: если open уже ниже sl, выходим по open
                    exit_price = min(cur_open, pos["sl"])
                    to_close.append((magic, exit_price, "SL"))
                elif cur_high >= pos["tp"]:
                    # Благоприятный гэп: если open выше tp, выходим по open
                    exit_price = max(cur_open, pos["tp"])
                    to_close.append((magic, exit_price, "TP"))
            else:
                if cur_high >= pos["sl"]:
                    exit_price = max(cur_open, pos["sl"])
                    to_close.append((magic, exit_price, "SL"))
                elif cur_low <= pos["tp"]:
                    exit_price = min(cur_open, pos["tp"])
                    to_close.append((magic, exit_price, "TP"))
                    
        for magic, price, reason in to_close:
            self.close_trade(magic, price, bar_time, reason)
            if reason == "SL":
                # Сохраняем время бара, а не глобальный счетчик
                self.last_sl_bar[magic] = bar_time
    
    def is_cooled(self, magic, cooldown, current_tf_df):
        if magic not in self.last_sl_bar:
            return False
        sl_time = self.last_sl_bar[magic]
        # Считаем бары именно этого таймфрейма после стопа
        bars_since = (current_tf_df.index > sl_time).sum()
        return bars_since < cooldown
    
    def bars_in_position(self, magic, current_tf_df):
        if magic not in self.positions:
            return 0
        entry_time = self.positions[magic]["bar"]
        # Считаем полные бары этого таймфрейма после входа
        return (current_tf_df.index > entry_time).sum()

def run():
    print("="*60)
    print("RSI LIVE SIMULATOR v4.1 - Honest Tester Edition")
    print("="*60)
    
    init_mt5()
    print("\nLoading data...")
    dfs = {}
    for tf_name, tf_data in CHAMPIONS.items():
        tf_val = tf_data["tf"]
        n = 30000 if tf_name == "M15" else 20000
        raw = load_data(tf_val, n)
        if raw is not None:
            df_clean = raw.dropna()
            dfs[tf_name] = prepare_indicators(df_clean, tf_data["rsi_period"])
            print(f"  {tf_name}: {len(dfs[tf_name])} bars (Indicators pre-calculated)")
    
    mt5.shutdown()
    
    if "M15" not in dfs:
        print("ERROR: Failed to load M15 data.")
        return

    test_days = 30
    sim = Simulator()
    
    df_m15 = dfs["M15"]
    split = df_m15.index[-1] - pd.Timedelta(days=test_days)
    timeline = df_m15[df_m15.index >= split].index
    
    print(f"\nTimeline: {len(timeline)} M15 bars")
    print("Simulating (v4.1: own-TF bar counts, gap-aware SL/TP, vectorized ind.)...")
    
    start_time = time.time()
    
    for bar_idx, bar_time in enumerate(timeline):
        if bar_idx % 500 == 0:
            elapsed = time.time() - start_time
            pct = bar_idx / len(timeline) * 100
            print(f"  {pct:.0f}% | Bar {bar_idx}/{len(timeline)} | Pos: {len(sim.positions)} | Pend: {len(sim.pending)} | Bal: ${sim.balance:.2f}")
        
        for tf_name, tf_data in CHAMPIONS.items():
            df = dfs[tf_name]
            available = df[df.index <= bar_time]
            if len(available) < tf_data["rsi_period"] + 5:
                continue
            
            cur_open = available["open"].iloc[-1]
            cur_high = available["high"].iloc[-1]
            cur_low = available["low"].iloc[-1]
            cur_close = available["close"].iloc[-1]
            
            rsi = available["rsi"].iloc[-1]
            rsi_prev = available["rsi_prev"].iloc[-1]
            atr = available["atr"].iloc[-1]
            
            if pd.isna(rsi) or pd.isna(rsi_prev) or pd.isna(atr):
                continue
            
            # 1. Execute pending entries at this bar's open
            sim.process_pending(cur_open, bar_time, tf_name)
            
            # 2. Check SL/TP (Gap-aware)
            sim.check_sl_tp(cur_open, cur_high, cur_low, bar_time)
            
            # 3. Champions logic
            for ch in tf_data["champs"]:
                magic = sim.get_magic(tf_name, ch["id"])
                
                if magic not in sim.positions and magic not in sim.pending:
                    if sim.is_cooled(magic, ch["cd"], available):
                        continue
                    
                    if rsi > ch["buy_thr"] and rsi_prev <= ch["buy_thr"]:
                        sim.pending[magic] = (1, atr, ch["sl"], ch["tp"], tf_name)
                    elif rsi < ch["sell_thr"] and rsi_prev >= ch["sell_thr"]:
                        sim.pending[magic] = (-1, atr, ch["sl"], ch["tp"], tf_name)
                
                elif magic in sim.positions:
                    pos = sim.positions[magic]
                    bars_held = sim.bars_in_position(magic, available)
                    
                    if pos["dir"] == 1:
                        if rsi > 70:
                            sim.close_trade(magic, cur_close, bar_time, "EXIT_OB")
                        # ВНИМАНИЕ: Если вы хотите строгий кроссовер вниз, замените на:
                        # elif bars_held >= 3 and rsi < 50 and rsi_prev >= 50:
                        elif bars_held >= 3 and rsi < 50:
                            sim.close_trade(magic, cur_close, bar_time, "EXIT_MR")
                    elif pos["dir"] == -1:
                        if rsi < 30:
                            sim.close_trade(magic, cur_close, bar_time, "EXIT_OS")
                        elif bars_held >= 3 and rsi > 50:
                            sim.close_trade(magic, cur_close, bar_time, "EXIT_MR")
    
    # Close remaining at own TF's last close
    for magic in list(sim.positions.keys()):
        pos = sim.positions[magic]
        tf_name = pos["tf_name"]
        if tf_name in dfs:
            last_close = dfs[tf_name]["close"].iloc[-1]
        else:
            last_close = dfs["M15"]["close"].iloc[-1]
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
        
        print(f"\n{'TF':<6} {'Trades':>8} {'PnL':>10} {'WR':>6}")
        print("-"*35)
        for tf_idx, tf_name in enumerate(CHAMPIONS.keys()):
            tf_trades = [t for t in sim.trades if MAGIC_BASE + tf_idx * 10 <= t[1] < MAGIC_BASE + (tf_idx+1) * 10]
            if tf_trades:
                tf_pnl = sum(t[0] for t in tf_trades)
                tf_wr = sum(1 for t in tf_trades if t[0] > 0) / len(tf_trades) * 100
                print(f"{tf_name:<6} {len(tf_trades):>8} {tf_pnl:>+10.1f} {tf_wr:>5.0f}%")
        
        print(f"\n{'Reason':<12} {'Count':>8} {'PnL':>10}")
        print("-"*35)
        for reason in ["SL", "TP", "EXIT_OB", "EXIT_OS", "EXIT_MR", "EOD"]:
            r_trades = [t for t in sim.trades if t[3] == reason]
            if r_trades:
                r_pnl = sum(t[0] for t in r_trades)
                print(f"{reason:<12} {len(r_trades):>8} {r_pnl:>+10.1f}")
    
    # Save safely
    save_dir = "D:\\PYTHON\\METATRADER"
    os.makedirs(save_dir, exist_ok=True)
    
    results = {"trades": len(sim.trades), "balance": sim.balance, "pnls": [t[0] for t in sim.trades]}
    with open(os.path.join(save_dir, "rsi_sim_results.json"), "w") as f:
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
        ax.set_title(f"RSI Simulator v4.1 - {len(sim.trades)} trades | ${sim.balance:.2f}")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Balance ($)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rsi_sim_equity.png"), dpi=150)
        print(f"\nSaved: rsi_sim_equity.png")
    except Exception as e:
        print(f"Plot error: {e}")

if __name__ == "__main__":
    run()