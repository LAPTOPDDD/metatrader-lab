import streamlit as st
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import lightgbm as lgb
from sklearn.linear_model import SGDRegressor
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# --- Константы ---
TF_MINUTES = {"5m": 5, "10m": 10, "15m": 15, "30m": 30, "60m": 60}
TF_COLORS = {"5m": "#3498db", "10m": "#e74c3c", "15m": "#2ecc71", "30m": "#9b59b6", "60m": "#e67e22"}
META_COLOR = "#FFD700"

# --- 1. Загрузка данных ---
@st.cache_data(ttl=300)
def load_mt5_data():
    if not mt5.initialize():
        st.error("Не удалось инициализировать MT5. Убедитесь, что терминал запущен.")
        return None
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 15000)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        st.error("Не получены данные от MT5.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    # Фильтр: только будни
    df = df[df.index.dayofweek < 5]
    return df

def resample_data(df_m5, tf_str):
    rule = tf_str.replace('m', 'T')
    df_res = df_m5.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
    }).dropna()
    return df_res

# --- 2. Инженерия признаков и таргета ---
def add_targets_features(df, sigma):
    df = df.copy()
    df['ma20'] = df['close'].rolling(20).mean()
    df['gauss'] = gaussian_filter1d(df['close'], sigma)
    df['delta_gauss'] = df['gauss'].diff()
    df['target'] = df['delta_gauss'].shift(-1)
    
    # 50 признаков (лаги от 1 до 10)
    for i in range(1, 11):
        df[f'gc_lag_{i}'] = df['delta_gauss'].shift(i)
        df[f'open_dif_{i}'] = df['open'].diff().shift(i)
        df[f'high_dif_{i}'] = df['high'].diff().shift(i)
        df[f'low_dif_{i}'] = df['low'].diff().shift(i)
        df[f'close_dif_{i}'] = df['close'].diff().shift(i)
    return df

# --- 3. Инициализация Session State ---
def init_session_state():
    defaults = {
        "step": 0,
        "predictions": {tf: [] for tf in TF_MINUTES},
        "meta_history": [],
        "meta_model": None,
        "meta_fitted": False,
        "meta_buffer": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# --- 4. Логика одного шага симуляции ---
def run_simulation_step(df_m5, params):
    test_size = 100
    train_bars_5m = params["train_bars_5m"]
    train_bars_tf = params["train_bars_tf"]
    sigma = params["sigma"]
    norm_mode = params["norm_mode"]
    meta_lr = params["meta_lr"]
    meta_alpha = params["meta_alpha"]

    step = st.session_state.step
    if step == 0:
        st.session_state.meta_model = SGDRegressor(
            loss="squared_error", learning_rate="constant", eta0=meta_lr,
            alpha=meta_alpha, penalty="l2", warm_start=True, max_iter=1, tol=None
        )
        st.session_state.meta_fitted = False
        st.session_state.meta_buffer = []
        
    # Текущий срез данных
    cut_idx = len(df_m5) - test_size
    current_m5 = df_m5.iloc[:cut_idx + step].copy()
    current_m5 = add_targets_features(current_m5, sigma)
    
    step_time = current_m5.index[-1]
    norm_preds = {}
    raw_preds = {}

    # Обучение базовых моделей
    for tf, mins in TF_MINUTES.items():
        if tf == "5m":
            df_tf = current_m5.copy()
            t_bars = train_bars_5m
        else:
            df_tf = resample_data(current_m5, tf)
            t_bars = train_bars_tf
            
        if len(df_tf) < t_bars + 20: continue
        
        train_df = df_tf.iloc[-t_bars:].dropna()
        if len(train_df) == 0: continue
        
        feat_cols = [c for c in train_df.columns if c.startswith(('gc_lag', 'open_dif', 'high_dif', 'low_dif', 'close_dif'))]
        X_train = train_df[feat_cols].values
        y_train = train_df['target'].values
        
        model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, verbose=-1)
        model.fit(X_train, y_train)
        
        # Прогноз на последней строке
        last_row = df_tf.iloc[[-1]]
        if last_row[feat_cols].isna().any().any(): continue
        
        X_pred = last_row[feat_cols].values
        pred = model.predict(X_pred)[0]
        raw_preds[tf] = pred
        
        # Нормализация
        if norm_mode == "Bar-normalized":
            norm_val = pred * 10000 / np.sqrt(mins / 5)
        elif norm_mode == "Пипсы":
            norm_val = pred * 10000
        else: # Raw
            norm_val = pred
            
        norm_preds[tf] = norm_val
        st.session_state.predictions[tf].append({
            "time": last_row.index[0],
            "step_time": step_time,
            "value": norm_val
        })

    # --- Мета-модель (Online Learning) ---
    if len(norm_preds) == 5:
        feat_order = ["5m", "10m", "15m", "30m", "60m"]
        X_k = np.array([[norm_preds[tf] for tf in feat_order]])
        
        # 1. Дообучение на фактах из буфера (появился факт для предыдущего шага)
        if len(st.session_state.meta_buffer) > 0:
            buf_item = st.session_state.meta_buffer.pop(0)
            # Ищем факт в текущем current_m5 по времени буфера
            buf_time = buf_item["step_time"]
            if buf_time in current_m5.index:
                y_true = current_m5.loc[buf_time, 'target']
                if not np.isnan(y_true):
                    st.session_state.meta_model.partial_fit(buf_item["features"], [y_true])
                    st.session_state.meta_fitted = True
                    # Обновляем историю с фактом
                    for h in reversed(st.session_state.meta_history):
                        if h["step"] == buf_item["step"]:
                            h["y_true"] = y_true
                            break

        # 2. Прогноз мета-модели
        if st.session_state.meta_fitted:
            meta_pred = st.session_state.meta_model.predict(X_k)[0]
        else:
            meta_pred = np.mean(X_k) # Холодный старт
            
        # 3. Сохраняем в буфер ожидания факта
        st.session_state.meta_buffer.append({
            "features": X_k,
            "step_time": step_time,
            "step": step
        })
        
        # 4. Записываем в историю
        st.session_state.meta_history.append({
            "time": step_time,
            "step": step,
            "meta_pred": meta_pred,
            "y_true": None, # Заполнится на следующем шаге
            "components": norm_preds.copy()
        })

# --- 5. Визуализация ---
def render_dashboard(df_m5, params):
    test_size = 100
    cut_idx = len(df_m5) - test_size
    sigma = params["sigma"]
    show_fact = params["show_fact"]
    show_meta = params["show_meta"]
    skip_first = params["skip_first"]
    
    # Подготовка полного датасета для отрисовки свечей
    plot_df = df_m5.copy()
    plot_df['ma20'] = plot_df['close'].rolling(20).mean()
    plot_df['gauss'] = gaussian_filter1d(plot_df['close'], sigma)
    plot_df['delta_gauss'] = plot_df['gauss'].diff()
    
    step_time = plot_df.index[cut_idx + st.session_state.step - 1] if st.session_state.step > 0 else plot_df.index[cut_idx]
    
    # Создание графиков
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.02)
    
    # 1. Свечной график
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df['open'], high=plot_df['high'],
        low=plot_df['low'], close=plot_df['close'], name="EURUSD M5",
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350', opacity=0.7
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ma20'], name="MA20", line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_vline(x=plot_df.index[cut_idx], line_width=1, line_dash="dash", line_color="yellow", row=1, col=1)
    
    # 2. График прогнозов
    if show_fact:
        fact_df = plot_df.iloc[cut_idx:]
        fig.add_trace(go.Scatter(x=fact_df.index, y=fact_df['delta_gauss'], name="Факт Δgauss 5m", 
                                 line=dict(color='white', width=1, dash='dot')), row=2, col=1)
                                 
    for tf in TF_MINUTES:
        preds = st.session_state.predictions[tf]
        if len(preds) > skip_first:
            p_df = pd.DataFrame(preds[skip_first:])
            fig.add_trace(go.Scatter(x=p_df['step_time'], y=p_df['value'], name=tf, 
                                     line_shape='hv', line=dict(color=TF_COLORS[tf], width=1.5)), row=2, col=1)
                                     
    if show_meta and len(st.session_state.meta_history) > 0:
        m_df = pd.DataFrame(st.session_state.meta_history[skip_first:])
        fig.add_trace(go.Scatter(x=m_df['time'], y=m_df['meta_pred'], name='META', 
                                 line_shape='hv', line=dict(color=META_COLOR, width=2.5),
                                 marker=dict(symbol='diamond', size=5)), row=2, col=1)
                                 
    fig.add_hline(y=0, line_width=1, line_color="gray", row=2, col=1)
    
    # Оформление
    fig.update_layout(template='plotly_dark', paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e', 
                      height=700, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Delta", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# --- 6. Метрики и Таблица ---
def render_metrics_and_table():
    meta_hist = st.session_state.meta_history
    valid_hist = [h for h in meta_hist if h['y_true'] is not None]
    
    if len(valid_hist) >= 5:
        y_true = np.array([h['y_true'] for h in valid_hist])
        y_pred = np.array([h['meta_pred'] for h in valid_hist])
        
        mae_pips = np.mean(np.abs(y_true - y_pred)) * 10000
        dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
        corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Обуч. примеров", len(valid_hist))
        c2.metric("MAE (pips)", f"{mae_pips:.1f}")
        c3.metric("Direction Accuracy", f"{dir_acc*100:.1f}%")
        c4.metric("Correlation", f"{corr:.3f}")
        
        with st.expander("Веса мета-модели (coef_)"):
            model = st.session_state.meta_model
            if hasattr(model, 'coef_'):
                coef_df = pd.DataFrame({
                    "TF": ["5m", "10m", "15m", "30m", "60m"],
                    "Weight": model.coef_
                })
                coef_df.loc[len(coef_df)] = ["Intercept", model.intercept_[0]]
                st.dataframe(coef_df.set_index("TF"), use_container_width=True)

    # Таблица сигналов
    if st.session_state.step > 0:
        table_data = []
        feat_order = ["5m", "10m", "15m", "30m", "60m"]
        last_meta = st.session_state.meta_history[-1] if st.session_state.meta_history else None
        
        for tf in feat_order:
            preds = st.session_state.predictions[tf]
            if preds:
                last_p = preds[-1]
                table_data.append({
                    "TF": tf,
                    "Time": last_p['time'],
                    "Value (Norm)": last_p['value'],
                    "Dir": "🟢 UP" if last_p['value'] > 0 else "🔴 DOWN"
                })
                
        if last_meta:
            table_data.append({
                "TF": "🧠 META",
                "Time": last_meta['time'],
                "Value (Norm)": last_meta['meta_pred'],
                "Dir": "🟢 UP" if last_meta['meta_pred'] > 0 else "🔴 DOWN"
            })
            
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# --- 7. Главный интерфейс ---
def main():
    st.set_page_config(page_title="MT5 Multi-TF Dashboard", layout="wide")
    
    with st.sidebar:
        st.title("⚙️ Параметры")
        train_bars_5m = st.slider("Train баров (5m)", 100, 1000, 300, step=50)
        train_bars_tf = st.slider("Train баров (10m-60m)", 50, 500, 150, step=10)
        sigma = st.slider("Sigma (Gauss)", 1, 20, 5)
        norm_mode = st.selectbox("Нормализация", ["Bar-normalized", "Пипсы", "Raw"])
        show_fact = st.checkbox("Показать факт Δgauss 5m", value=True)
        show_meta = st.checkbox("Показать мета-сигнал", value=True)
        skip_first = st.slider("Skip first N точек", 0, 10, 1)
        meta_lr = st.slider("Meta learning rate", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
        meta_alpha = st.slider("Meta L2 alpha", 1e-5, 1e-2, 1e-4, format="%.5f")
        
        params = {
            "train_bars_5m": train_bars_5m, "train_bars_tf": train_bars_tf,
            "sigma": sigma, "norm_mode": norm_mode, "show_fact": show_fact,
            "show_meta": show_meta, "skip_first": skip_first,
            "meta_lr": meta_lr, "meta_alpha": meta_alpha
        }
        
        st.markdown("---")
        st.subheader("Симуляция")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➡", use_container_width=True):
                if st.session_state.step < 100:
                    st.session_state.step += 1
                    run_simulation_step(df_m5, params)
                    
        with col2:
            if st.button("⏩ +10", use_container_width=True):
                for _ in range(10):
                    if st.session_state.step < 100:
                        st.session_state.step += 1
                        run_simulation_step(df_m5, params)
                        
        with col3:
            if st.button("🔁 Сброс", use_container_width=True):
                st.session_state.step = 0
                st.session_state.predictions = {tf: [] for tf in TF_MINUTES}
                st.session_state.meta_history = []
                st.session_state.meta_fitted = False
                st.session_state.meta_buffer = []

    st.title("📈 Мульти-Таймфреймный дашборд EURUSD (MT5 + LightGBM + SGD)")
    st.caption(f"Текущий шаг: {st.session_state.step} / 100")
    
    if df_m5 is not None:
        render_dashboard(df_m5, params)
        render_metrics_and_table()
    else:
        st.warning("Ожидание данных...")

if __name__ == "__main__":
    init_session_state()
    df_m5 = load_mt5_data()
    main()