"""
Test filling modes - find which one works with this broker
"""
import MetaTrader5 as mt5

mt5.initialize()
mt5.symbol_select("EURUSD", True)

FILLING_MODES = [
    ("ORDER_FILLING_RETURN", mt5.ORDER_FILLING_RETURN),
    ("ORDER_FILLING_IOC", mt5.ORDER_FILLING_IOC),
    ("ORDER_FILLING_FOK", mt5.ORDER_FILLING_FOK),
]

tick = mt5.symbol_info_tick("EURUSD")
price = tick.ask

for name, mode in FILLING_MODES:
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "EURUSD",
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": 0,
        "tp": 0,
        "magic": 999999,
        "comment": "TEST",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mode,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"WINNER: {name}")
        # Close it
        pos = mt5.positions_get(magic=999999)
        if pos:
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": "EURUSD",
                "volume": 0.01,
                "type": mt5.ORDER_TYPE_SELL,
                "price": tick.bid,
                "position": pos[0].ticket,
                "magic": 999999,
                "comment": "TEST_CL",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mode,
            }
            mt5.order_send(close_request)
        break
    else:
        print(f"FAILED: {name} -> {result.comment if result else 'no result'}")

mt5.shutdown()
