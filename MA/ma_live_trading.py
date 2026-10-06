"""
MA Ensemble - Live Trading via MT5 Python API
V9 Champions (corrected entry) | M15+M30+H1 | cooldown=3
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time, json, datetime

# ============================================================
# CONFIG
# ============================================================
SYMBOL = "EURUSD"
LOT = 0.01
COOLDOWN = 3
CHECK_INTERVAL = 5  # seconds

MAGIC_BASE = 200000
CHAMPIONS = {
    "M15": {
        "tf": mt5.TIMEFRAME_M15,
        "champs": [
            {"id":0, "fast":26, "slow":44, "entry":0.483, "sl":0.50, "tp":3.51},
            {"id":1, "fast":25, "slow":44, "entry":0.483, "sl":0.50, "tp":3.39},
            {"id":2, "fast":24, "slow":46, "entry":0.418, "sl":0.50, "tp":3.40},
            {"id":3, "fast":26, "slow":45, "entry":0.358, "sl":0.50, "tp":3.45},
            {"id":4, "fast":26, "slow":42, "entry":0.599, "sl":0.50, "tp":3.32},
        ]
    },
    "M30": {
        "tf": mt5.TIMEFRAME_M30,
        "champs": [
            {"id":0, "fast":49, "slow":60, "entry":0.957, "sl":0.56, "tp":3.01},
            {"id":1, "fast":49, "slow":60, "entry":0.922, "sl":0.56, "tp":3.01},
            {"id":2, "fast":49, "slow":60, "entry":0.919, "sl":0.57, "tp":3.06},
            {"id":3, "fast":48, "slow":59, "entry":0.903, "sl":0.57, "tp":3.01},
            {"id":4, "fast":48, "slow":59, "entry":0.949, "sl":0.57, "tp":3.00},
        ]
    },
    "H1": {
        "tf": mt5.TIMEFRAME_H1,
        "champs": [
            {"id":0, "fast":69, "slow":80, "entry":1.118, "sl":0.67, "tp":0.69},
            {"id":1, "fast":67, "slow":80, "entry":1.175, "sl":0.56, "tp":3.00},
            {"id":2, "fast":69, "slow":80, "entry":1.164, "sl":0.50, "tp":2.93},
            {"id":3, "fast":69, "slow":80, "entry":1.162, "sl":0.53, "tp":2.94},
            {"id":4, "fast":68, "slow":80, "entry":1.021, "sl":0.59, "tp":0.60},
        ]
    }
}

# ============================================================
# STATE
# ============================================================
last_sl_bar = {}  # {magic: bar_time}
last_bar_time = 0
running = True

# ============================================================
# FUNCTIONS
# ============================================================
def get_magic(tf_name, ch_id):
    tf_idx = list(CHAMPIONS.keys()).index(tf_name)
    return MAGIC_BASE + tf_idx * 10 + ch_id

def calc_sma(symbol, tf, period, shift=1):
    rates = mt5.copy_rates_from_pos(symbol, tf, shift, period)
    if rates is None or len(rates) < period:
        return None
    closes = [r[4] for r in rates]  # close
    return sum(closes) / period

def calc_atr(symbol, tf, shift=1):
    rates = mt5.copy_rates_from_pos(symbol, tf, shift, 15)
    if rates is None or len(rates) < 15:
        return None
    closes = [r[4] for r in rates]
    highs = [r[2] for r in rates]
    lows = [r[3] for r in rates]
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    if len(trs) < 14:
        return None
    return sum(trs[-14:]) / 14

def get_position_type(magic):
    for i in range(mt5.positions_total()):
        pos = mt5.positions_get(index=i)
        if pos and pos[0].symbol == SYMBOL and pos[0].magic == magic:
            return 1 if pos[0].type == mt5.ORDER_TYPE_BUY else -1
    return 0

def open_trade(magic, direction, atr, sl_mult, tp_mult):
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False
    
    dig = mt5.symbol_info(SYMBOL).digits
    if direction == 1:  # BUY
        price = tick.ask
        sl = round(price - atr * sl_mult, dig)
        tp = round(price + atr * tp_mult, dig)
        order_type = mt5.ORDER_TYPE_BUY
    else:  # SELL
        price = tick.bid
        sl = round(price + atr * sl_mult, dig)
        tp = round(price - atr * tp_mult, dig)
        order_type = mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": magic,
        "comment": "V9_Py",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return True
    return False

def close_position(magic):
    for i in range(mt5.positions_total()):
        pos = mt5.positions_get(index=i)
        if pos and pos[0].symbol == SYMBOL and pos[0].magic == magic:
            close_type = mt5.ORDER_TYPE_SELL if pos[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(SYMBOL)
            price = tick.bid if pos[0].type == mt5.ORDER_TYPE_BUY else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": pos[0].volume,
                "type": close_type,
                "price": price,
                "position": pos[0].ticket,
                "magic": magic,
                "comment": "V9_Py_close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE
    return False

# ============================================================
# MAIN LOOP
# ============================================================
def tick():
    global last_bar_time
    
    # Get current M5 bar time for new-bar check
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 1)
    if rates is None or len(rates) == 0:
        return
    cur_bar = rates[0][0]  # time
    if cur_bar == last_bar_time:
        return
    last_bar_time = cur_bar
    
    for tf_name, tf_data in CHAMPIONS.items():
        tf = tf_data["tf"]
        atr = calc_atr(SYMBOL, tf)
        if atr is None or atr <= 0:
            continue
        
        for ch in tf_data["champs"]:
            magic = get_magic(tf_name, ch["id"])
            
            maf = calc_sma(SYMBOL, tf, ch["fast"])
            mas = calc_sma(SYMBOL, tf, ch["slow"])
            if maf is None or mas is None:
                continue
            
            diff = (maf - mas) / atr
            pt = get_position_type(magic)
            
            if pt == 0:  # no position
                # Check cooldown
                cooled = False
                if magic in last_sl_bar:
                    bars_since = (cur_bar - last_sl_bar[magic]) // 300
                    if bars_since < COOLDOWN:
                        cooled = True
                
                if diff > ch["entry"] and not cooled:
                    if open_trade(magic, 1, atr, ch["sl"], ch["tp"]):
                        print("%s BUY %s ch=%d diff=%.3f" % (datetime.datetime.now(), tf_name, ch["id"], diff))
                
                elif diff < -ch["entry"] and not cooled:
                    if open_trade(magic, -1, atr, ch["sl"], ch["tp"]):
                        print("%s SELL %s ch=%d diff=%.3f" % (datetime.datetime.now(), tf_name, ch["id"], diff))
            
            else:  # has position
                if pt == 1 and diff < -ch["entry"] * 0.3:
                    if close_position(magic):
                        print("%s CLOSE_BUY %s ch=%d" % (datetime.datetime.now(), tf_name, ch["id"]))
                elif pt == -1 and diff > ch["entry"] * 0.3:
                    if close_position(magic):
                        print("%s CLOSE_SELL %s ch=%d" % (datetime.datetime.now(), tf_name, ch["id"]))

def check_sl():
    """Check for SL hits in history"""
    global last_sl_bar
    deals = mt5.history_deals_get(datetime.datetime.now() - datetime.timedelta(hours=1), datetime.datetime.now())
    if deals:
        for deal in deals:
            if deal.symbol == SYMBOL and deal.reason == mt5.DEAL_REASON_SL:
                last_sl_bar[deal.magic] = deal.time

def main():
    global running
    
    if not mt5.initialize():
        print("MT5 init failed")
        return
    
    print("="*60)
    print("MA Ensemble Live Trading")
    print("V9 Champions | M15+M30+H1 | Cooldown=%d" % COOLDOWN)
    print("Symbol: %s | Lot: %.2f" % (SYMBOL, LOT))
    print("="*60)
    
    # Check account
    account = mt5.account_info()
    print("Account: %d | Balance: $%.2f | Leverage: 1:%d" % (account.login, account.balance, account.leverage))
    
    while running:
        try:
            check_sl()
            tick()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print("Error:", e)
            time.sleep(10)
    
    mt5.shutdown()
    print("Stopped")

if __name__ == "__main__":
    main()
