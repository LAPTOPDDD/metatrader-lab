"""
MA V7 - Multi-Symbol Comparison
All major FX + Gold | walk-forward | equity + table
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

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "USDCNH", "USDSEK", "XAUUSD",
]
TF_CONFIG = {
    "M5":  (mt5.TIMEFRAME_M5,  50000, 40),
    "M15": (mt5.TIMEFRAME_M15, 30000, 50),
    "M30": (mt5.TIMEFRAME_M30, 20000, 60),
    "H1":  (mt5.TIMEFRAME_H1,  20000, 80),
}
N_TRIALS   = 80
TRAIN_DAYS = 60
TEST_DAYS  = 30
STEP_DAYS  = 30
COST_PER_TRADE = 0.7  # pips: spread + commission + slippage

def get_pip_size(symbol):
    info = mt5.symbol_info(symbol)
    if info is None: return 0.0001
    return info.point * 10 if info.digits == 5 else info.point

def load_data(symbol, tf, n_bars):
    try: mt5.shutdown()
    except: pass
    if not mt5.initialize():
        raise RuntimeError(f"MT5: {mt5.last_error()}")
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown(); raise RuntimeError(f"No {symbol}")
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
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

def run_fast(o, h, l, c, atr, fast, slow, entry, sl_atr, tp_atr, pip_size, cost):
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
                if   l[i]<=sl: trades.append(-(sl-ep)/pip_size - cost); pos=0
                elif h[i]>=tp: trades.append((tp-ep)/pip_size - cost);  pos=0
                elif d<-entry*0.3 or (b>=3 and d<=0): trades.append((c[i]-ep)/pip_size - cost); pos=0
            else:
                if   h[i]>=sl: trades.append((ep-sl)/pip_size - cost);  pos=0
                elif l[i]<=tp: trades.append((ep-tp)/pip_size - cost);  pos=0
                elif d>entry*0.3 or (b>=3 and d>=0): trades.append((ep-c[i])/pip_size - cost); pos=0
    return np.array(trades) if trades else np.array([])

def objective(trial, o, h, l, c, atr, max_slow, pip_size, cost):
    fast = trial.suggest_int("fast",3,max(3,max_slow-11))
    slow = trial.suggest_int("slow",fast+10,max_slow)
    entry = trial.suggest_float("entry",0.05,1.2)
    sl = trial.suggest_float("sl_atr",0.5,4.0)
    tp = trial.suggest_float("tp_atr",0.5,4.0)
    pnl = run_fast(o,h,l,c,atr,fast,slow,entry,sl,tp,pip_size,cost)
    if len(pnl)<5: return -9999
    wr=(pnl>0).mean()
    sh=pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else -9
    return sh*(1+wr-0.5)*np.sign(pnl.sum()+0.01)

def run_symbol(symbol, pip_size):
    results_per_tf = {}
    equity_per_tf = {}

    for tf_name, (tf_val, n_bars, max_slow) in TF_CONFIG.items():
        df = load_data(symbol, tf_val, n_bars)
        if df is None or len(df) < 1000:
            continue

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

        if not windows: continue

        all_oos = []
        tf_results = []

        for w, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            df_train = df[(df.index >= tr_s) & (df.index < tr_e)]
            df_test  = df[(df.index >= te_s) & (df.index < te_e)]
            if len(df_train) < 500 or len(df_test) < 50: continue

            o,h,l,c,atr = precompute(df_train)
            o2,h2,l2,c2,atr2 = precompute(df_test)

            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
            study.optimize(lambda t: objective(t,o,h,l,c,atr,max_slow,pip_size,COST_PER_TRADE),
                           n_trials=N_TRIALS, show_progress_bar=False)

            trials = sorted([t for t in study.trials if t.value is not None and t.value>-9990],
                            key=lambda t:t.value, reverse=True)

            window_pnl = []
            for t in trials[:5]:
                p = t.params
                pnl_t = run_fast(o2,h2,l2,c2,atr2,**p,pip_size=pip_size,cost=COST_PER_TRADE)
                window_pnl.append(pnl_t)

            if window_pnl:
                max_len = max(len(x) for x in window_pnl)
                combined = np.zeros(max_len)
                for pnl in window_pnl:
                    combined[:len(pnl)] += pnl
                all_oos.append(combined)

                n = len(combined)
                wr = (combined>0).mean() if n>0 else 0
                sh = combined.mean()/combined.std()*np.sqrt(252) if n>5 and combined.std()>0 else 0
                tot = combined.sum()
                tf_results.append({"window":w+1,"trades":n,"wr":wr,"sharpe":sh,"pips":tot})

        if all_oos:
            max_total = max(len(x) for x in all_oos)
            total = np.zeros(max_total)
            for pnl in all_oos:
                total[:len(pnl)] += pnl
            results_per_tf[tf_name] = tf_results
            equity_per_tf[tf_name] = total

    # Combine all TFs
    if equity_per_tf:
        max_all = max(len(x) for x in equity_per_tf.values())
        combined_all = np.zeros(max_all)
        for pnl in equity_per_tf.values():
            combined_all[:len(pnl)] += pnl
        return results_per_tf, equity_per_tf, combined_all
    return None, None, None

if __name__ == "__main__":
    try: mt5.shutdown()
    except: pass

    print("="*70)
    print("MA V7 - Multi-Symbol Comparison")
    print(f"10 symbols | M5+M15+M30+H1 | walk-forward | cost={COST_PER_TRADE} pips/trade")
    print("="*70)

    all_data = {}

    for symbol in SYMBOLS:
        try: mt5.shutdown()
        except: pass
        if not mt5.initialize():
            continue
        if not mt5.symbol_select(symbol, True):
            mt5.shutdown(); continue
        info = mt5.symbol_info(symbol)
        mt5.shutdown()

        pip_size = info.point * 10 if info.digits == 5 else info.point
        print(f"\n{'='*50}")
        print(f"  {symbol} | digits={info.digits} pip={pip_size}")
        print(f"{'='*50}")

        tf_res, equity_tf, combined = run_symbol(symbol, pip_size)

        if tf_res:
            # Summarize
            total_pips = sum(sum(r["pips"] for r in res) for res in tf_res.values())
            total_trades = sum(sum(r["trades"] for r in res) for res in tf_res.values())
            profitable_windows = sum(sum(1 for r in res if r["pips"]>0) for res in tf_res.values())
            total_windows = sum(len(res) for res in tf_res.values())
            avg_wr = np.mean([r["wr"] for res in tf_res.values() for r in res])
            avg_sh = np.mean([r["sharpe"] for res in tf_res.values() for r in res])

            all_data[symbol] = {
                "pip_size": pip_size,
                "total_pips": total_pips,
                "total_trades": total_trades,
                "profitable_windows": profitable_windows,
                "total_windows": total_windows,
                "win_pct": profitable_windows/total_windows if total_windows>0 else 0,
                "avg_wr": avg_wr,
                "avg_sharpe": avg_sh,
                "equity": combined,
                "per_tf": {tf: sum(r["pips"] for r in res) for tf, res in tf_res.items()},
            }

            print(f"\n  RESULT: {profitable_windows}/{total_windows} profitable "
                  f"| pips={total_pips:+.0f} | trades={total_trades} "
                  f"| wr={avg_wr:.0%} | sharpe={avg_sh:.1f}")
            for tf, pips in all_data[symbol]["per_tf"].items():
                print(f"    {tf:>4}: {pips:+.0f} pips")
        else:
            print(f"\n  SKIP - not enough data")

    # Table
    print(f"\n{'='*90}")
    print("FINAL RANKING")
    print(f"{'='*90}")

    ranking = sorted(all_data.items(), key=lambda x: x[1]["total_pips"], reverse=True)

    print(f"\n{'#':>2} {'Symbol':>8} {'Pips':>8} {'Trades':>7} {'WinWin':>7} {'AvgWR':>6} {'AvgSH':>7} {'M5':>8} {'M15':>8} {'M30':>8} {'H1':>8}")
    print("-"*90)
    for i, (sym, d) in enumerate(ranking):
        tf_strs = {tf: d["per_tf"].get(tf, 0) for tf in ["M5","M15","M30","H1"]}
        print(f"{i+1:>2} {sym:>8} {d['total_pips']:>+8.0f} {d['total_trades']:>7} "
              f"{d['profitable_windows']}/{d['total_windows']:<4} "
              f"{d['avg_wr']:>5.0%} {d['avg_sharpe']:>7.1f} "
              f"{tf_strs['M5']:>+8.0f} {tf_strs['M15']:>+8.0f} "
              f"{tf_strs['M30']:>+8.0f} {tf_strs['H1']:>+8.0f}")

    # Save table
    table_data = []
    for sym, d in ranking:
        table_data.append({
            "Symbol": sym,
            "Total Pips": d["total_pips"],
            "Trades": d["total_trades"],
            "Win Windows": f"{d['profitable_windows']}/{d['total_windows']}",
            "Avg WR": f"{d['avg_wr']:.0%}",
            "Avg Sharpe": round(d["avg_sharpe"],1),
            "M5 Pips": d["per_tf"].get("M5",0),
            "M15 Pips": d["per_tf"].get("M15",0),
            "M30 Pips": d["per_tf"].get("M30",0),
            "H1 Pips": d["per_tf"].get("H1",0),
        })
    pd.DataFrame(table_data).to_csv("D:\\PYTHON\\METATRADER\\ma_v7_ranking.csv", index=False)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Equity per symbol
    ax = axes[0]
    cmap = plt.cm.tab10
    for i, (sym, d) in enumerate(ranking):
        if d["equity"] is not None:
            cum = np.cumsum(d["equity"])
            ax.plot(cum, label=f"{sym} ({d['total_pips']:+.0f})", color=cmap(i), lw=1.5, alpha=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("Cumulative Pips (after costs)")
    ax.set_title("Walk-Forward Equity by Symbol (M5+M15+M30+H1 ensemble)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Bar chart
    ax = axes[1]
    syms = [s for s, _ in ranking]
    pips = [d["total_pips"] for _, d in ranking]
    colors = ["#2ca02c" if p>0 else "#d62728" for p in pips]
    bars = ax.barh(syms[::-1], pips[::-1], color=colors[::-1], alpha=0.7)
    ax.axvline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Total Pips (after costs)")
    ax.set_title("Symbol Ranking")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("D:\\PYTHON\\METATRADER\\ma_v7_multi.png", dpi=150)
    print(f"\nSaved: ma_v7_multi.png + ma_v7_ranking.csv")
