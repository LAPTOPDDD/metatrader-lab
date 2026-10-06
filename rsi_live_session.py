"""
RSI Ensemble - Live Trading with Session Filter
15 Champions | M15+M30+H1 | Session: 08:00-17:00
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time, datetime

# ============================================================
# CONFIG
# ============================================================
SYMBOL = "EURUSD"
LOT = 0.01
CHECK_INTERVAL = 5

SESSION_START = 8
SESSION_END = 17

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
# STATE
# ============================================================
last_sl_bar = {}
last_rsi = {}
last_bar_time = 0
running = True
trades_today = 0

# ============================================================
# FUNCTIONS
# ============================================================
def in_session():
    now = datetime.datetime.now()
    hour = now.hour
    if SESSION_START <= SESSION_END:
        return SESSION_START <= hour < SESSION_END
    else:
        return hour >= SESSION_START or hour < SESSION_END

def get_magic(tf_name, ch_id):
    tf_idx = list(CHAMPIONS.keys()).index(tf_name)
    return MAGIC_BASE + tf_idx * 10 + ch_id

def calc_rsi(symbol, tf, period):
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, period + 50)
    if rates is None or len(rates) < period + 10:
        return None
    closes = [r[4] for r in rates]
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_rsi_prev(symbol, tf, period):
    rates = mt5.copy_rates_from_pos(symbol, tf, 1, period + 50)
    if rates is None or len(rates) < period + 10:
        return None
    closes = [r[4] for r in rates]
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr(symbol, tf):
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 15)
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

def count_positions():
    count = 0
    for i in range(mt5.positions_total()):
        pos = mt5.positions_get(index=i)
        if pos and pos[0].symbol == SYMBOL and pos[0].magic >= MAGIC_BASE:
            count += 1
    return count

def open_trade(magic, direction, atr, sl_mult, tp_mult):
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False
    
    dig = mt5.symbol_info(SYMBOL).digits
    if direction == 1:
        price = tick.ask
        sl = round(price - atr * sl_mult, dig)
        tp = round(price + atr * tp_mult, dig)
        order_type = mt5.ORDER_TYPE_BUY
    else:
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
        "comment": "RSI_Sess",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return True
    print("  ORDER FAILED: %s" % result.comment if result else "no result")
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
                "comment": "RSI_Sess_cl",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE
    return False

def close_all():
    for i in range(mt5.positions_total()):
        pos = mt5.positions_get(index=i)
        if pos and pos[0].symbol == SYMBOL and pos[0].magic >= MAGIC_BASE:
            close_position(pos[0].magic)

def tick():
    global last_bar_time, trades_today
    
    if not in_session():
        return
    
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 1)
    if rates is None or len(rates) == 0:
        return
    cur_bar = rates[0][0]
    if cur_bar == last_bar_time:
        return
    last_bar_time = cur_bar
    
    for tf_name, tf_data in CHAMPIONS.items():
        tf = tf_data["tf"]
        period = tf_data["rsi_period"]
        atr = calc_atr(SYMBOL, tf)
        if atr is None or atr <= 0:
            continue
        
        rsi_now = calc_rsi(SYMBOL, tf, period)
        rsi_prev = calc_rsi_prev(SYMBOL, tf, period)
        if rsi_now is None or rsi_prev is None:
            continue
        
        for ch in tf_data["champs"]:
            magic = get_magic(tf_name, ch["id"])
            pt = get_position_type(magic)
            
            if pt == 0:
                cooled = False
                if magic in last_sl_bar:
                    bars_since = (cur_bar - last_sl_bar[magic]) // 300
                    if bars_since < ch["cd"]:
                        cooled = True
                
                # Buy: RSI crosses above buy_thr (oversold bounce)
                if rsi_now > ch["buy_thr"] and rsi_prev <= ch["buy_thr"] and not cooled:
                    if open_trade(magic, 1, atr, ch["sl"], ch["tp"]):
                        trades_today += 1
                        print("%s BUY %s ch=%d RSI=%.1f->%.1f" % (
                            datetime.datetime.now().strftime("%H:%M"), tf_name, ch["id"], rsi_prev, rsi_now))
                
                # Sell: RSI crosses below sell_thr (overbought reversal)
                elif rsi_now < ch["sell_thr"] and rsi_prev >= ch["sell_thr"] and not cooled:
                    if open_trade(magic, -1, atr, ch["sl"], ch["tp"]):
                        trades_today += 1
                        print("%s SELL %s ch=%d RSI=%.1f->%.1f" % (
                            datetime.datetime.now().strftime("%H:%M"), tf_name, ch["id"], rsi_prev, rsi_now))
            
            else:
                # Exit rules
                if pt == 1 and (rsi_now > 70 or rsi_now < 50):
                    if close_position(magic):
                        print("%s CLOSE_BUY %s ch=%d RSI=%.1f" % (
                            datetime.datetime.now().strftime("%H:%M"), tf_name, ch["id"], rsi_now))
                elif pt == -1 and (rsi_now < 30 or rsi_now > 50):
                    if close_position(magic):
                        print("%s CLOSE_SELL %s ch=%d RSI=%.1f" % (
                            datetime.datetime.now().strftime("%H:%M"), tf_name, ch["id"], rsi_now))

def check_sl():
    global last_sl_bar
    deals = mt5.history_deals_get(datetime.datetime.now() - datetime.timedelta(hours=1), datetime.datetime.now())
    if deals:
        for deal in deals:
            if deal.symbol == SYMBOL and deal.reason == mt5.DEAL_REASON_SL:
                last_sl_bar[deal.magic] = deal.time

def end_of_day():
    pos = count_positions()
    if pos > 0:
        print("Session end - closing all %d positions" % pos)
        close_all()

def main():
    global running, trades_today
    
    if not mt5.initialize():
        print("MT5 init failed")
        return
    
    print("="*60)
    print("RSI Ensemble Live - Session Filter")
    print("15 Champions | M15+M30+H1")
    print("Session: %02d:00-%02d:00 (server time)" % (SESSION_START, SESSION_END))
    print("Symbol: %s | Lot: %.2f" % (SYMBOL, LOT))
    print("="*60)
    
    account = mt5.account_info()
    print("Account: %d | Balance: $%.2f" % (account.login, account.balance))
    print("")
    
    session_active = False
    
    while running:
        try:
            now_active = in_session()
            
            if now_active and not session_active:
                trades_today = 0
                print("-"*60)
                print("SESSION STARTED at %s" % datetime.datetime.now().strftime("%H:%M:%S"))
                print("-"*60)
                session_active = True
            
            if not now_active and session_active:
                end_of_day()
                print("SESSION ENDED at %s | Trades: %d" % (datetime.datetime.now().strftime("%H:%M:%S"), trades_today))
                print("")
                session_active = False
            
            if now_active:
                check_sl()
                tick()
            
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print("Error:", e)
            time.sleep(10)
    
    close_all()
    mt5.shutdown()
    print("Stopped - all positions closed")

if __name__ == "__main__":
    main()
