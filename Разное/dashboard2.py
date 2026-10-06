import streamlit as st
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import PassiveAggressiveRegressor, Ridge
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="EURUSD Multi-TF Dashboard", layout="wide")

# ────────────────────────────────────────────
# Данные
# ────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_mt5_data(n_bars: int = 15000) -> pd.DataFrame:
    if not mt5.initialize():
        st.error(f"MT5 initialize() failed: {mt5.last_error()}"); st.stop()
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, n_bars)
    if rates is None or len(rates) == 0:
        st.error("Не удалось получить данные из MT5"); mt5.shutdown(); st.stop()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df[df.index.dayofweek < 5]
    mt5.shutdown()
    return df[["open", "high", "low", "close", "tick_volume"]].copy()


def resample_ohlc(df_m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df_m5.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum",
    }).dropna()


def add_features(df: pd.DataFrame, sigma: int = 5, n_lags: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["gauss"] = gaussian_filter1d(df["close"].values, sigma=sigma)
    df["delta_gauss"] = df["gauss"].diff()
    df["target"] = df["delta_gauss"].shift(-1)
    for i in range(1, n_lags + 1):
        df[f"gc_lag_{i}"] = df["delta_gauss"].shift(i)
        df[f"open_dif_{i}"] = df["open"].diff().shift(i)
        df[f"high_dif_{i}"] = df["high"].diff().shift(i)
        df[f"low_dif_{i}"] = df["low"].diff().shift(i)
        df[f"close_dif_{i}"] = df["close"].diff().shift(i)
    df["ma20"] = df["close"].rolling(20).mean()
    return df


def get_feature_cols(n_lags: int = 10) -> list:
    cols = []
    for i in range(1, n_lags + 1):
        cols.append(f"gc_lag_{i}")
        for p in ("open_dif", "high_dif", "low_dif", "close_dif"):
            cols.append(f"{p}_{i}")
    return cols


def train_and_predict(df: pd.DataFrame, train_size: int, feature_cols: list):
    df_clean = df.dropna(subset=feature_cols + ["target"])
    if len(df_clean) < train_size + 1:
        return None
    train = df_clean.iloc[-train_size - 1: -1]
    last_row = df_clean.iloc[[-1]]
    if len(train) < 20:
        return None
    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        verbose=-1, n_jobs=-1,
    )
    model.fit(train[feature_cols], train["target"])
    return float(model.predict(last_row[feature_cols])[0])


# ────────────────────────────────────────────
# Нормализация
# ────────────────────────────────────────────
TF_MINUTES = {"5m": 5, "10m": 10, "15m": 15, "30m": 30, "60m": 60}

def normalize_pred(val, tf_name, mode):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if mode == "Raw":
        return val
    if mode == "Пипсы (×10000)":
        return val * 10000
    if mode == "Bar-normalized":
        return val * 10000 / np.sqrt(TF_MINUTES[tf_name] / 5)
    return val

def to_meta_units(val_raw, tf_name):
    return val_raw * 10000 / np.sqrt(TF_MINUTES[tf_name] / 5)


# ────────────────────────────────────────────
# Мета-модели
# ────────────────────────────────────────────
class EMAEnsembleMeta:
    """
    Отслеживает EMA ошибки |y - x_i| для каждого TF.
    Веса ~ 1/EMA_error (нормализованы), прогноз = взвешенная сумма.
    Всегда даёт нетривиальный прогноз с амплитудой факта.
    """
    def __init__(self, n_features: int, alpha: float = 0.15):
        self.alpha = alpha
        self.err_ema = np.ones(n_features) * 1.0  # начальная ошибка
        self.n_updates = 0

    def predict(self, x: np.ndarray) -> float:
        w = 1.0 / (self.err_ema + 1e-6)
        w = w / w.sum()
        return float(np.dot(w, x))

    def partial_fit(self, x: np.ndarray, y: float):
        err = np.abs(x - y)
        self.err_ema = (1 - self.alpha) * self.err_ema + self.alpha * err
        self.n_updates += 1

    def get_weights(self) -> np.ndarray:
        w = 1.0 / (self.err_ema + 1e-6)
        return w / w.sum()


class RollingRidgeMeta:
    """
    Ridge регрессия на скользящем окне последних N (X, y) + пост-масштабирование
    (чтобы std(pred) ≈ std(y_true) — убирает shrinkage).
    """
    def __init__(self, window: int = 40, alpha: float = 1.0):
        self.window = window
        self.alpha = alpha
        self.X_buf = []
        self.y_buf = []
        self.model = None
        self.scale = 1.0

    def partial_fit(self, x: np.ndarray, y: float):
        self.X_buf.append(x.copy())
        self.y_buf.append(float(y))
        if len(self.X_buf) > self.window:
            self.X_buf = self.X_buf[-self.window:]
            self.y_buf = self.y_buf[-self.window:]

        if len(self.X_buf) >= 8:
            X_arr = np.array(self.X_buf)
            y_arr = np.array(self.y_buf)
            self.model = Ridge(alpha=self.alpha, fit_intercept=True)
            self.model.fit(X_arr, y_arr)
            # Пост-масштабирование
            preds = self.model.predict(X_arr)
            std_p = preds.std() + 1e-9
            std_y = y_arr.std() + 1e-9
            self.scale = std_y / std_p

    @property
    def n_updates(self):
        return len(self.X_buf)

    def predict(self, x: np.ndarray) -> float:
        if self.model is None:
            return float(np.mean(x))
        raw = float(self.model.predict(x.reshape(1, -1))[0])
        # Приведение к «правильной» амплитуде
        y_mean = np.mean(self.y_buf) if self.y_buf else 0.0
        return y_mean + self.scale * (raw - y_mean)

    def get_weights(self) -> np.ndarray:
        if self.model is None:
            return np.ones(5) / 5
        return self.model.coef_


class PAMeta:
    """Passive-Aggressive + пост-масштабирование."""
    def __init__(self, C: float = 0.5):
        self.model = PassiveAggressiveRegressor(
            C=C, max_iter=1, warm_start=True, tol=None, random_state=42,
        )
        self.fitted = False
        self.n_updates = 0
        self.y_hist = []
        self.pred_hist = []
        self.scale = 1.0

    def partial_fit(self, x: np.ndarray, y: float):
        X = x.reshape(1, -1); yy = np.array([y])
        self.model.partial_fit(X, yy)
        self.fitted = True
        self.n_updates += 1
        self.y_hist.append(y)
        if len(self.y_hist) > 60:
            self.y_hist = self.y_hist[-60:]
            self.pred_hist = self.pred_hist[-60:]
        # обновляем scale
        if len(self.y_hist) >= 8 and len(self.pred_hist) >= 8:
            std_y = np.std(self.y_hist) + 1e-9
            std_p = np.std(self.pred_hist) + 1e-9
            self.scale = std_y / std_p

    def predict(self, x: np.ndarray) -> float:
        if not self.fitted:
            return float(np.mean(x))
        raw = float(self.model.predict(x.reshape(1, -1))[0])
        self.pred_hist.append(raw)
        y_mean = np.mean(self.y_hist) if self.y_hist else 0.0
        return y_mean + self.scale * (raw - y_mean)

    def get_weights(self):
        if not self.fitted:
            return np.ones(5) / 5
        return self.model.coef_


# ────────────────────────────────────────────
# Конфиг
# ────────────────────────────────────────────
TF_CONFIG = {
    "5m":  {"rule": None,    "color": "#1f77b4", "symbol": "🔵"},
    "10m": {"rule": "10min", "color": "#d62728", "symbol": "🔴"},
    "15m": {"rule": "15min", "color": "#2ca02c", "symbol": "🟢"},
    "30m": {"rule": "30min", "color": "#9467bd", "symbol": "🟣"},
    "60m": {"rule": "60min", "color": "#8c564b", "symbol": "🟤"},
}
TF_NAMES = list(TF_CONFIG.keys())
FEATURE_COLS = get_feature_cols(10)

# ────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────
st.sidebar.header("⚙️ Настройки")
train_5m = st.sidebar.slider("Train баров (5m)", 100, 1000, 300, step=50)
train_other = st.sidebar.slider("Train баров (10m–60m)", 50, 500, 150, step=25)
sigma = st.sidebar.slider("Sigma (Gauss)", 1, 20, 5)
norm_mode = st.sidebar.selectbox(
    "Нормализация предсказаний",
    ["Bar-normalized", "Пипсы (×10000)", "Raw"],
)
show_truth = st.sidebar.checkbox("Показать факт (истинный Δgauss 5m)", True)
show_meta = st.sidebar.checkbox("Показать мета-сигнал", True)
skip_first = st.sidebar.slider("Скрыть первые N точек прогноза", 0, 10, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Мета-модель")
meta_type = st.sidebar.selectbox(
    "Тип мета-модели",
    ["EMA-Ensemble", "Rolling Ridge", "Passive-Aggressive"],
    index=0,
    help="EMA-Ensemble — самый стабильный на малой выборке",
)
if meta_type == "EMA-Ensemble":
    ema_alpha = st.sidebar.slider("EMA α (скорость адаптации)", 0.05, 0.5, 0.15, 0.05)
elif meta_type == "Rolling Ridge":
    ridge_window = st.sidebar.slider("Окно (баров)", 15, 100, 40, 5)
    ridge_alpha = st.sidebar.select_slider("Ridge α", options=[0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)
else:
    pa_C = st.sidebar.select_slider("PA C", options=[0.01, 0.05, 0.1, 0.5, 1.0], value=0.5)

cold_start_bars = st.sidebar.slider("Холодный старт (баров)", 3, 30, 5)

test_bars = 100
train_sizes = {"5m": train_5m, **{tf: train_other for tf in TF_NAMES[1:]}}

# ────────────────────────────────────────────
# Данные
# ────────────────────────────────────────────
raw_m5 = load_mt5_data(15000)
split_idx = len(raw_m5) - test_bars
train_pool = raw_m5.iloc[:split_idx].copy()
test_pool = raw_m5.iloc[split_idx:].copy()

# ────────────────────────────────────────────
# Фабрика меты
# ────────────────────────────────────────────
def init_meta():
    if meta_type == "EMA-Ensemble":
        return EMAEnsembleMeta(n_features=5, alpha=ema_alpha)
    elif meta_type == "Rolling Ridge":
        return RollingRidgeMeta(window=ridge_window, alpha=ridge_alpha)
    else:
        return PAMeta(C=pa_C)

# ────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0
if "predictions" not in st.session_state:
    st.session_state.predictions = {tf: [] for tf in TF_NAMES}
if "meta_history" not in st.session_state:
    st.session_state.meta_history = []
if "meta_model" not in st.session_state:
    st.session_state.meta_model = init_meta()
    st.session_state.meta_type_cached = meta_type
if "meta_buffer" not in st.session_state:
    st.session_state.meta_buffer = []

# Если пользователь поменял тип меты — пересоздаём и чистим историю
if st.session_state.get("meta_type_cached") != meta_type:
    st.session_state.meta_model = init_meta()
    st.session_state.meta_type_cached = meta_type
    st.session_state.meta_history = []
    st.session_state.meta_buffer = []
    st.session_state.step = 0
    st.session_state.predictions = {tf: [] for tf in TF_NAMES}

# ────────────────────────────────────────────
# Кнопки
# ────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
with c1:
    if st.button("Следующий бар ➡️"):
        if st.session_state.step < len(test_pool):
            st.session_state.step += 1
with c2:
    if st.button("+10 ⏩"):
        st.session_state.step = min(st.session_state.step + 10, len(test_pool))
with c3:
    if st.button("Сброс 🔁"):
        st.session_state.step = 0
        st.session_state.predictions = {tf: [] for tf in TF_NAMES}
        st.session_state.meta_history = []
        st.session_state.meta_model = init_meta()
        st.session_state.meta_buffer = []

current_step = st.session_state.step
with c4:
    st.caption(
        f"Шаг: **{current_step}** / {len(test_pool)}  •  σ={sigma}  •  "
        f"Meta: **{meta_type}**  •  обучено: {st.session_state.meta_model.n_updates}"
    )

# ────────────────────────────────────────────
# Текущий датафрейм
# ────────────────────────────────────────────
if current_step > 0:
    current_m5 = pd.concat([train_pool, test_pool.iloc[:current_step]])
else:
    current_m5 = train_pool.copy()

tf_dataframes = {}
for tf_name, cfg in TF_CONFIG.items():
    df_tf = current_m5.copy() if cfg["rule"] is None else resample_ohlc(current_m5, cfg["rule"])
    tf_dataframes[tf_name] = add_features(df_tf, sigma=sigma, n_lags=10)

# ────────────────────────────────────────────
# Catch-up loop
# ────────────────────────────────────────────
if current_step > 0:
    preds_history = st.session_state.predictions
    expected_len = current_step

    while any(len(preds_history[tf]) < expected_len for tf in TF_NAMES):
        done = len(preds_history["5m"])
        target_step = done + 1
        step_time = test_pool.index[target_step - 1]

        m5_slice = pd.concat([train_pool, test_pool.iloc[:target_step]])

        for tf_name, cfg in TF_CONFIG.items():
            if len(preds_history[tf_name]) >= target_step:
                continue
            df_tf = m5_slice.copy() if cfg["rule"] is None else resample_ohlc(m5_slice, cfg["rule"])
            df_tf = add_features(df_tf, sigma=sigma, n_lags=10)
            mask = df_tf.index <= step_time
            pred = train_and_predict(df_tf[mask], train_sizes[tf_name], FEATURE_COLS) if mask.any() else None
            last_time = df_tf[mask].index[-1] if mask.any() else step_time
            preds_history[tf_name].append({
                "time": last_time, "step_time": step_time,
                "delta_gauss": pred, "step": target_step,
            })

        # ─── МЕТА ───
        # Дообучение на факте
        if len(st.session_state.meta_buffer) > 0:
            df_5m_now = add_features(m5_slice, sigma=sigma, n_lags=10)
            new_buffer = []
            for item in st.session_state.meta_buffer:
                idx_arr = df_5m_now.index.get_indexer([item["step_time"]])
                idx = idx_arr[0]
                if idx >= 0 and idx + 1 < len(df_5m_now):
                    y_true_raw = df_5m_now["delta_gauss"].iloc[idx + 1]
                    if pd.notna(y_true_raw):
                        y_true_norm = to_meta_units(y_true_raw, "5m")
                        X = np.array(item["features"], dtype=float)
                        try:
                            st.session_state.meta_model.partial_fit(X, y_true_norm)
                        except Exception as e:
                            st.warning(f"Meta fit error: {e}")
                        for h in st.session_state.meta_history:
                            if h["step"] == item["step"]:
                                h["y_true"] = float(y_true_norm)
                                break
                    else:
                        new_buffer.append(item)
                else:
                    new_buffer.append(item)
            st.session_state.meta_buffer = new_buffer

        # Прогноз меты
        feat_vec = []
        valid = True
        for tf in TF_NAMES:
            v = preds_history[tf][-1]["delta_gauss"]
            if v is None:
                valid = False; break
            feat_vec.append(to_meta_units(v, tf))

        if valid:
            X_meta = np.array(feat_vec, dtype=float)
            if st.session_state.meta_model.n_updates < cold_start_bars:
                weights = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
                meta_pred = float(np.dot(feat_vec, weights))
            else:
                try:
                    meta_pred = float(st.session_state.meta_model.predict(X_meta))
                except Exception:
                    meta_pred = float(np.mean(feat_vec))

            st.session_state.meta_history.append({
                "time": step_time, "step": target_step,
                "meta_pred": meta_pred, "y_true": None,
                "components": feat_vec,
            })
            st.session_state.meta_buffer.append({
                "features": feat_vec,
                "step_time": step_time,
                "step": target_step,
            })

    st.session_state.predictions = preds_history

# ────────────────────────────────────────────
# Визуализация
# ────────────────────────────────────────────
df_5m = tf_dataframes["5m"]
display_window = 300
df_display = df_5m.iloc[-display_window:] if len(df_5m) > display_window else df_5m

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.02, row_heights=[0.65, 0.35],
)

fig.add_trace(go.Candlestick(
    x=df_display.index, open=df_display["open"], high=df_display["high"],
    low=df_display["low"], close=df_display["close"],
    name="EURUSD M5", opacity=0.7,
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["ma20"], mode="lines",
    name="MA20", line=dict(color="orange", width=1.5),
), row=1, col=1)

if current_step > 0:
    split_time = test_pool.index[0]
    if split_time >= df_display.index[0]:
        fig.add_vline(x=split_time, line_dash="dash",
                      line_color="yellow", opacity=0.5, row=1, col=1)

fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)

for tf_name, cfg in TF_CONFIG.items():
    preds = st.session_state.predictions.get(tf_name, [])
    if len(preds) > skip_first:
        preds_use = preds[skip_first:]
        df_pred = pd.DataFrame(preds_use).dropna(subset=["delta_gauss"])
        if len(df_pred):
            df_pred["norm"] = df_pred["delta_gauss"].apply(
                lambda v: normalize_pred(v, tf_name, norm_mode)
            )
            fig.add_trace(go.Scatter(
                x=df_pred["time"], y=df_pred["norm"],
                mode="lines+markers",
                name=f"{cfg['symbol']} {tf_name}",
                line=dict(color=cfg["color"], width=1.3, shape="hv"),
                marker=dict(size=4), opacity=0.7,
            ), row=2, col=1)

if show_truth and current_step > 0:
    df_truth = df_5m[["delta_gauss"]].dropna().iloc[-min(display_window, current_step + 20):]
    df_truth = df_truth.copy()
    df_truth["norm"] = df_truth["delta_gauss"].apply(
        lambda v: normalize_pred(v, "5m", norm_mode)
    )
    fig.add_trace(go.Scatter(
        x=df_truth.index, y=df_truth["norm"],
        mode="lines", name="✅ Факт Δgauss 5m",
        line=dict(color="white", width=1, dash="dot"), opacity=0.6,
    ), row=2, col=1)

if show_meta and st.session_state.meta_history:
    df_meta = pd.DataFrame(st.session_state.meta_history)
    if len(df_meta) > skip_first:
        df_meta = df_meta.iloc[skip_first:]
        # meta_pred хранится в bar-normalized единицах 5m (= пипсы 5m)
        if norm_mode == "Raw":
            df_meta["meta_show"] = df_meta["meta_pred"] / 10000
        else:
            df_meta["meta_show"] = df_meta["meta_pred"]

        fig.add_trace(go.Scatter(
            x=df_meta["time"], y=df_meta["meta_show"],
            mode="lines+markers", name="🧠 META (online)",
            line=dict(color="#FFD700", width=2.8, shape="hv"),
            marker=dict(size=7, symbol="diamond"),
        ), row=2, col=1)

fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
fig.update_layout(
    height=780, margin=dict(l=50, r=20, t=10, b=30),
    plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
    font=dict(color="#d4d4d4", size=11),
    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0, font=dict(size=10)),
    yaxis=dict(title="Price", gridcolor="#333"),
    yaxis2=dict(title=f"Δgauss ({norm_mode})", gridcolor="#333"),
)
st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────
# Метрики
# ────────────────────────────────────────────
if st.session_state.meta_history:
    df_m = pd.DataFrame(st.session_state.meta_history).dropna(subset=["y_true"])
    if len(df_m) >= 5:
        y_true = df_m["y_true"].values
        y_pred = df_m["meta_pred"].values
        mae = float(np.mean(np.abs(y_true - y_pred)))
        dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
        std_ratio = float(np.std(y_pred) / (np.std(y_true) + 1e-9))

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Обучено", int(st.session_state.meta_model.n_updates))
        m2.metric("MAE", f"{mae:.3f}")
        m3.metric("Dir Acc", f"{dir_acc*100:.1f}%")
        m4.metric("Corr", f"{corr:.3f}")
        m5.metric("Std ratio", f"{std_ratio:.2f}",
                  help="std(pred)/std(true). Хорошо ~0.7-1.2. Меньше 0.3 = shrinkage")

        try:
            weights = st.session_state.meta_model.get_weights()
            df_w = pd.DataFrame({
                "TF": TF_NAMES,
                "Weight": [f"{c:+.4f}" for c in np.array(weights).flatten()[:5]],
            })
            with st.expander(f"🧠 Веса {meta_type}"):
                st.dataframe(df_w, use_container_width=True)
        except Exception:
            pass

# ────────────────────────────────────────────
# Таблица сигналов
# ────────────────────────────────────────────
st.subheader("📊 Текущие сигналы")

signal_data = []
for tf_name in TF_NAMES:
    preds = st.session_state.predictions.get(tf_name, [])
    if preds:
        last = preds[-1]; v = last["delta_gauss"]
        signal_data.append({
            "TF": tf_name,
            "Time": str(last["time"]),
            "Δgauss (raw)": f"{v:.6f}" if v is not None else "—",
            "Norm": f"{normalize_pred(v, tf_name, norm_mode):.3f}" if v is not None else "—",
            "Dir": ("🟢 UP" if v and v > 0 else "🔴 DOWN") if v else "—",
        })
if st.session_state.meta_history:
    last_meta = st.session_state.meta_history[-1]
    signal_data.append({
        "TF": "🧠 META",
        "Time": str(last_meta["time"]),
        "Δgauss (raw)": "—",
        "Norm": f"{last_meta['meta_pred']:.3f}",
        "Dir": "🟢 UP" if last_meta["meta_pred"] > 0 else "🔴 DOWN",
    })

if signal_data:
    st.dataframe(pd.DataFrame(signal_data).set_index("TF"), use_container_width=True)
else:
    st.info("Нажмите «Следующий бар ➡️», чтобы сгенерировать первые сигналы.")

st.caption(
    f"MT5 Multi-TF LightGBM • M5: {len(raw_m5)} • "
    f"Train: {len(train_pool)} • Test: {len(test_pool)}"
)