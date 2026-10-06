import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import time
import datetime as dt
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== ПАРАМЕТРЫ ====================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M30
BARS_HISTORY = 13000
SIGMA = 2
LOOKAHEAD = int(4 * SIGMA)
PIP = 0.0001
LOT = 0.01              # безопасный лот для демо
M = 5                   # ретрейн каждые 5 баров
N = 3                   # удержание N баров
THRESH = 0.50           # порог уверенности
SL_PIPS = 20            # стоп-лосс в пунктах (None = без стопа)
DRY_RUN = False         # True = только лог, без ордеров
MAX_BARS = 0            # >0 = обработать N баров и выйти (для проверки)
POLL_SEC = 5            # частота опроса нового бара

OUT_FILE = 'ver_3_equity_live.csv'

# ==================== MT5 ====================
if not mt5.initialize():
    sys.exit("MT5 init failed")
acc = mt5.account_info()
print(f"Account: {acc.login} | server={acc.server} | balance=${acc.balance:.2f} | "
      f"equity=${acc.equity:.2f} | trade_allowed={acc.trade_allowed}")

# ==================== ФИЧИ (КАУЗАЛЬНЫЕ) ====================
def add_features(d):
    d = d.copy()
    d['MACD'] = d['close'].ewm(span=12).mean() - d['close'].ewm(span=26).mean()
    for i in range(1, 6):
        d[f'MACD_lag_{i}'] = d['MACD'].shift(i)
    for k in [3, 6, 12, 24]:
        d[f'mom_{k}'] = d['close'].pct_change(k)
    delta = d['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
    rs = gain / loss.replace(0, np.nan)
    d['rsi_14'] = 100 - 100 / (1 + rs)
    tr = np.maximum(d['high'] - d['low'],
                    np.maximum(abs(d['high'] - d['close'].shift()),
                               abs(d['low'] - d['close'].shift())))
    d['atr_14'] = tr.ewm(alpha=1/14).mean()
    for span in [20, 50, 100]:
        d[f'ema{span}_pos'] = d['close'] / d['close'].ewm(span=span).mean() - 1
    for k in [10, 20]:
        d[f'vol_{k}'] = d['close'].rolling(k).std()
    rng = (d['high'] - d['low']).replace(0, np.nan)
    d['body'] = (d['close'] - d['open']) / rng
    d['upper_shadow'] = (d['high'] - np.maximum(d['open'], d['close'])) / rng
    d['lower_shadow'] = (np.minimum(d['open'], d['close']) - d['low']) / rng
    d['hl_range'] = (d['high'] - d['low']) / d['close']
    d['close_open'] = (d['close'] - d['open']) / d['close']
    d['vol_norm'] = d['tick_volume'] / d['tick_volume'].rolling(20).mean()
    return d


def build_model(df, end_excl):
    sub = add_features(df.iloc[:end_excl])
    sub['Conv_Deriv'] = gaussian_filter1d(sub['close'].values, SIGMA, order=1)
    sub['Target_Binary'] = np.where(sub['Conv_Deriv'] > 0, 1, -1)
    sub.dropna(inplace=True)
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03,
                           num_leaves=31, min_child_samples=20,
                           verbosity=-1, random_state=42, n_jobs=1)
    m.fit(sub[FEATS], sub['Target_Binary'])
    return m


def get_feats(df):
    return [c for c in df.columns if c not in ('time', 'open', 'high', 'low', 'close',
                                               'tick_volume', 'spread', 'real_volume',
                                               'bar_idx', 'Conv_Deriv', 'Target_Binary')]


# ==================== ОРДЕРА ====================
def open_position(side):
    """side: mt5.ORDER_TYPE_BUY / SELL. Возвращает ticket или None."""
    if DRY_RUN:
        print(f"  [DRY] OPEN {side}")
        return 12345
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("  ! no tick", flush=True)
        return None
    price = tick.ask if side == mt5.ORDER_TYPE_BUY else tick.bid
    sl = 0.0
    if SL_PIPS:
        sl = price - SL_PIPS * PIP if side == mt5.ORDER_TYPE_BUY else price + SL_PIPS * PIP
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": side,
        "price": price,
        "sl": sl,
        "tp": 0.0,
        "deviation": 20,
        "magic": 777001,
        "comment": "ver_3_live",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  ! open failed: {result.retcode} {result.comment}", flush=True)
        return None
    print(f"  [LIVE] OPEN {'BUY' if side == mt5.ORDER_TYPE_BUY else 'SELL'} @ {price} sl={sl} "
          f"ticket={result.order}", flush=True)
    return result.order


def close_position(position):
    if DRY_RUN:
        print(f"  [DRY] CLOSE pos {position.ticket}")
        return True
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False
    side = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = tick.bid if side == mt5.ORDER_TYPE_SELL else tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": side,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": 777001,
        "comment": "ver_3_live_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    ok = result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        print(f"  [LIVE] CLOSE @ {price} ticket={position.ticket}", flush=True)
    else:
        print(f"  ! close failed: {result.retcode} {result.comment}", flush=True)
    return ok


def get_position():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions:
        return positions[0]
    return None


# ==================== ИСТОРИЯ ====================
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_HISTORY)
if rates is None or len(rates) == 0:
    sys.exit("No history")
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df = df[~df.index.dayofweek.isin([5, 6])]
df = add_features(df)
FEATS = get_feats(df)

# Модель: обучаем на всей истории с отступом от последнего закрытого бара
last_ts = df.index[-1]
model = build_model(df, len(df) - LOOKAHEAD)
print(f"Initial model trained on {len(df)} bars, last bar: {last_ts}")

# ==================== LIVE ЦИКЛ ====================
# Состояние
my_pos = get_position()          # позиция, открытая нашей стратегией
held = 0
processed_ts = last_ts
bar_count = 0
np.random.seed(42)

if OUT_FILE and not os.path.exists(OUT_FILE):
    open(OUT_FILE, 'w').write("time,close,pred,pos,equity\n")

import os

def append_equity(ts, close, pred, pos, equity):
    with open(OUT_FILE, 'a') as f:
        f.write(f"{ts},{close},{pred},{pos},{equity:.2f}\n")

print("Live loop started. Waiting for new bars...", flush=True)
try:
    while True:
        # ---- Ждём закрытие нового бара ----
        new_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 2)
        if new_rates is None or len(new_rates) < 2:
            time.sleep(POLL_SEC)
            continue
        last_closed = new_rates[-2]  # предпоследний = последний закрытый бар
        ts = dt.datetime.utcfromtimestamp(last_closed['time'])
        if ts <= processed_ts:
            time.sleep(POLL_SEC)
            continue
        processed_ts = ts
        bar_count += 1
        print(f"\n--- NEW BAR {ts} ({bar_count}) ---", flush=True)

        # Догружаем историю и пересчитываем
        all_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_HISTORY + bar_count)
        df = pd.DataFrame(all_rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[~df.index.dayofweek.isin([5, 6])]
        df = add_features(df)
        df = df.loc[:ts]

        # Ретрейн каждые M баров
        if bar_count % M == 0 and bar_count > 0:
            model = build_model(df, len(df) - LOOKAHEAD)
            print(f"  [RETRAIN] model refitted on {len(df)} bars", flush=True)

        # Предсказание по закрытому бару
        row = df.iloc[-1]
        proba = model.predict_proba(row[FEATS].to_frame().T)[0]
        p1 = proba[1]
        pred = 1 if p1 >= THRESH else (-1 if p1 <= 1 - THRESH else 0)
        print(f"  P(long)={p1:.3f} -> pred={pred}", flush=True)

        # ---- Управление позицией ----
        price = row['close']
        cur_pos = get_position()

        # Позиция была, но закрыта стопом/вручную
        if my_pos is not None and cur_pos is None:
            print(f"  ! position closed externally (SL hit?)", flush=True)
            held = 0
            my_pos = None

        if my_pos is not None:
            held += 1
            # Закрытие по N барам
            if held >= N:
                if close_position(my_pos):
                    my_pos = None
                    held = 0

        # Открытие новой позиции
        if my_pos is None and pred != 0:
            my_pos = open_position(mt5.ORDER_TYPE_BUY if pred == 1 else mt5.ORDER_TYPE_SELL)
            held = 0

        # Эквити
        acc = mt5.account_info()
        if DRY_RUN:
            append_equity(ts, price, pred, 1 if my_pos else 0, acc.equity)
        else:
            pos_val = 1 if get_position() else 0
            append_equity(ts, price, pred, pos_val, acc.equity)
        print(f"  equity=${acc.equity:.2f}", flush=True)

        if MAX_BARS and bar_count >= MAX_BARS:
            print(f"Reached MAX_BARS={MAX_BARS}, exiting", flush=True)
            break

except KeyboardInterrupt:
    print("\nStopped by user")

mt5.shutdown()
print("Done")
