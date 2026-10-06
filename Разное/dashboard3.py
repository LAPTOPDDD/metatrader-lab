import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb
from sklearn.linear_model import SGDRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================
# КОНФИГУРАЦИЯ И ЗАГРУЗКА ДАННЫХ
# ============================================

@st.cache_data(ttl=300)
def load_mt5_data():
    """Загрузка данных из MT5 или генерация синтетики"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise Exception("MT5 init failed")
        
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 15000)
        mt5.shutdown()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 
                               'close': 'Close', 'tick_volume': 'Volume'})
        df = df[df['time'].dt.dayofweek < 5]  # Только будни
        return df.sort_values('time').reset_index(drop=True), False
    except:
        # Синтетические данные для тестирования
        dates = pd.date_range(end=datetime.now(), periods=5000, freq='5min')
        dates = dates[dates.dayofweek < 5]
        price = 1.08 + np.cumsum(np.random.randn(len(dates)) * 0.0002)
        df = pd.DataFrame({
            'time': dates, 'Open': price, 'High': price * (1 + np.random.rand(len(dates)) * 0.0005),
            'Low': price * (1 - np.random.rand(len(dates)) * 0.0005), 'Close': price,
            'Volume': np.random.randint(100, 1000, len(dates))
        })
        return df.reset_index(drop=True), True

def resample_ohlc(df_m5, timeframe_min):
    """Ресемплинг M5 -> старшие ТФ"""
    rule = f'{timeframe_min}min'
    ohlc = df_m5.resample(rule, on='time').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna().reset_index()
    return ohlc

# ============================================
# ГЕНЕРАЦИЯ ПРИЗНАКОВ (БЕЗ УТЕЧЕК)
# ============================================

def add_features(df, sigma):
    """Добавление признаков с shift(1) для защиты от утечек"""
    df = df.copy()
    
    # Гауссово сглаживание
    df['gc'] = gaussian_filter1d(df['Close'].values, sigma=sigma)
    df['delta_gauss'] = df['gc'].diff()
    
    # Лаги (все через shift(1))
    for i in range(1, 11):
        df[f'gc_lag_{i}'] = df['gc'].shift(i)
        df[f'open_dif_{i}'] = df['Open'].diff().shift(i)
        df[f'high_dif_{i}'] = df['High'].diff().shift(i)
        df[f'low_dif_{i}'] = df['Low'].diff().shift(i)
        df[f'close_dif_{i}'] = df['Close'].diff().shift(i)
    
    # Доп. признак из ТЗ
    df['ma20'] = df['Close'].rolling(20).mean().shift(1)
    
    return df.dropna()

# ============================================
# ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================

def train_base_model(df, n_bars):
    """Полное переобучение LightGBM на последних n_bars"""
    feature_cols = [c for c in df.columns if '_lag_' in c or '_dif_' in c or c == 'ma20']
    X = df[feature_cols].iloc[-n_bars:]
    y = df['delta_gauss'].shift(-1).iloc[-n_bars:-1]  # Цель на 1 бар вперед
    
    if len(X) < 10 or len(y) < 10:
        return None, None
    
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, verbose=-1)
    model.fit(X.iloc[:-1], y.iloc[:-1])
    pred = model.predict(X.iloc[-1:])
    
    return model, pred[0]

# ============================================
# МЕТА-МОДЕЛЬ (ONLINE)
# ============================================

def normalize_prediction(pred, tf_minutes):
    """Bar-normalization: приведение к масштабу 5-минутного бара"""
    return pred * 10000 / np.sqrt(tf_minutes / 5.0)

def update_meta_model(meta_model, buffer, fitted_flag):
    """Инкрементальное обучение SGDRegressor"""
    if len(buffer) < 2:
        return meta_model, fitted_flag, None
    
    # Берем предыдущий шаг для обучения
    prev_step = buffer[-2]
    X = np.array(prev_step['features']).reshape(1, -1)
    y = prev_step['y_true']
    
    if y is None:
        return meta_model, fitted_flag, None
        
    if not fitted_flag:
        meta_model.partial_fit(X, np.array([y]))
        fitted_flag = True
    else:
        meta_model.partial_fit(X, np.array([y]))
    
    # Предсказание для текущего шага
    current_X = np.array(buffer[-1]['features']).reshape(1, -1)
    meta_pred = meta_model.predict(current_X)[0]
    
    return meta_model, fitted_flag, meta_pred

# ============================================
# UI ИНТЕРФЕЙС
# ============================================

st.set_page_config(page_title="EURUSD Multi-TF Dashboard", layout="wide")
st.title("📊 Multi-Timeframe Dashboard: EURUSD + Online Meta-Learning")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Параметры")
    data_source = st.radio("Источник данных", ["MT5 Live", "Синтетика"], index=0)
    
    sigma = st.slider("Sigma (Gauss)", 1, 20, 5)
    train_5m = st.slider("Train баров (5m)", 100, 1000, 300)
    train_higher = st.slider("Train баров (10m-60m)", 50, 500, 150)
    
    norm_mode = st.selectbox("Нормализация", ["Bar-normalized", "Raw"], index=0)
    show_fact = st.checkbox("Показать факт Δgauss", True)
    show_meta = st.checkbox("Показать Meta-сигнал", True)
    skip_n = st.slider("Skip first N точек", 0, 10, 1)
    
    st.subheader("Meta SGD")
    eta0 = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.4f")
    alpha = st.slider("L2 Alpha", 1e-5, 1e-2, 1e-4, format="%.5f")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("➕1 Бар"):
            st.session_state.step += 1
    with col2:
        if st.button("➕10 Баров"):
            st.session_state.step += 10
    with col3:
        if st.button("🔄 Сброс"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# --- Инициализация состояния ---
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.predictions = {tf: [] for tf in ['5m', '10m', '15m', '30m', '60m']}
    st.session_state.meta_history = []
    st.session_state.meta_buffer = []
    st.session_state.meta_model = SGDRegressor(
        loss="squared_error", learning_rate="constant", eta0=eta0,
        alpha=alpha, penalty="l2", warm_start=True, max_iter=1, tol=None
    )
    st.session_state.meta_fitted = False

# --- Загрузка данных ---
raw_df, is_synthetic = load_mt5_data()
if is_synthetic and data_source != "Синтетика":
    st.warning("⚠️ MT5 недоступен. Используются синтетические данные.")
elif data_source == "Синтетика":
    raw_df, _ = load_mt5_data()  # Принудительно используем синтетику

# --- Подготовка данных ---
tf_list = [('5m', 5), ('10m', 10), ('15m', 15), ('30m', 30), ('60m', 60)]
dfs = {}
for name, mins in tf_list:
    base_df = raw_df if mins == 5 else resample_ohlc(raw_df, mins)
    dfs[name] = add_features(base_df, sigma)

# --- Симуляция ---
current_step = min(st.session_state.step, len(dfs['5m']) - 10)
if current_step > 0:
    features_for_meta = []
    
    # Обучаем базовые модели
    for i, (name, mins) in enumerate(tf_list):
        df_tf = dfs[name]
        n_train = train_5m if mins == 5 else train_higher
        
        if current_step >= n_train:
            model, pred = train_base_model(df_tf.iloc[:current_step], n_train)
            if pred is not None:
                norm_pred = normalize_prediction(pred, mins) if norm_mode == "Bar-normalized" else pred
                features_for_meta.append(norm_pred)
                
                st.session_state.predictions[name].append({
                    'time': df_tf.iloc[current_step-1]['time'],
                    'step': current_step,
                    'pred': norm_pred,
                    'fact': df_tf.iloc[current_step]['delta_gauss'] if current_step < len(df_tf) else None
                })
        else:
            features_for_meta.append(0.0)
    
    # Обновляем мета-модель
    if len(features_for_meta) == 5:
        st.session_state.meta_buffer.append({
            'features': features_for_meta,
            'step_time': dfs['5m'].iloc[current_step-1]['time'],
            'y_true': None
        })
        
        # Устанавливаем факт для предыдущего шага
        if len(st.session_state.meta_buffer) > 1:
            prev_time = st.session_state.meta_buffer[-2]['step_time']
            fact_value = dfs['5m'][dfs['5m']['time'] == prev_time]['delta_gauss'].values
            if len(fact_value) > 0:
                st.session_state.meta_buffer[-2]['y_true'] = fact_value[0]
        
        # Обучение и предсказание
        meta_model, fitted, meta_pred = update_meta_model(
            st.session_state.meta_model, 
            st.session_state.meta_buffer,
            st.session_state.meta_fitted
        )
        st.session_state.meta_model = meta_model
        st.session_state.meta_fitted = fitted
        
        if meta_pred is not None:
            st.session_state.meta_history.append({
                'time': dfs['5m'].iloc[current_step-1]['time'],
                'pred': meta_pred,
                'fact': st.session_state.meta_buffer[-2].get('y_true')
            })

# ============================================
# ВИЗУАЛИЗАЦИЯ
# ============================================

fig = go.Figure()

# График цены (верхняя часть)
fig.add_trace(go.Scatter(
    x=dfs['5m']['time'], y=dfs['5m']['Close'],
    mode='lines', name='EURUSD Close', line=dict(color='white', width=1),
    yaxis='y1'
))

# Нулевая линия
fig.add_hline(y=0, line=dict(color='gray', dash='dash', width=0.5))

# Сигналы базовых моделей
colors = {'5m': '#FFFFFF', '10m': '#FF6B6B', '15m': '#4ECDC4', '30m': '#45B7D1', '60m': '#96CEB4'}
for name, color in colors.items():
    if st.session_state.predictions[name]:
        preds = st.session_state.predictions[name][skip_n:]
        if preds:
            fig.add_trace(go.Scatter(
                x=[p['time'] for p in preds],
                y=[p['pred'] for p in preds],
                mode='lines+markers' if name == '5m' else 'lines',
                name=f'{name} Signal',
                line=dict(color=color, width=1.5 if name == '5m' else 1, dash='dot' if name == '5m' else None),
                yaxis='y2'
            ))

# Мета-сигнал
if show_meta and st.session_state.meta_history:
    meta_data = st.session_state.meta_history[skip_n:]
    if meta_data:
        fig.add_trace(go.Scatter(
            x=[m['time'] for m in meta_data],
            y=[m['pred'] for m in meta_data],
            mode='lines+markers',
            name='META Signal',
            line=dict(color='#FFD700', width=2.5),
            marker=dict(symbol='diamond', size=6),
            yaxis='y2'
        ))

# Факт Δgauss
if show_fact and current_step > 0:
    fact_slice = dfs['5m'].iloc[max(0, current_step-200):current_step]
    fig.add_trace(go.Scatter(
        x=fact_slice['time'], y=fact_slice['delta_gauss'],
        mode='lines', name='Fact Δgauss',
        line=dict(color='lime', width=1), opacity=0.7, yaxis='y2'
    ))

# Настройка осей
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
    height=800,
    yaxis=dict(title="Price", domain=[0.65, 1]),
    yaxis2=dict(title="Signals", domain=[0, 0.6], zeroline=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================
# МЕТРИКИ И ТАБЛИЦЫ
# ============================================

if len(st.session_state.meta_history) >= 5:
    meta_df = pd.DataFrame(st.session_state.meta_history)
    valid = meta_df.dropna(subset=['fact'])
    
    if len(valid) >= 5:
        mae = np.mean(np.abs(valid['pred'] - valid['fact']))
        dir_acc = np.mean((valid['pred'] > 0) == (valid['fact'] > 0))
        corr = valid['pred'].corr(valid['fact'])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.6f}")
        m2.metric("Direction Acc", f"{dir_acc:.2%}")
        m3.metric("Correlation", f"{corr:.3f}")
        
        with st.expander("Веса мета-модели (coef_) - доверие к TF"):
            if hasattr(st.session_state.meta_model, 'coef_'):
                coef_df = pd.DataFrame({
                    'TF': list(colors.keys()),
                    'Weight': st.session_state.meta_model.coef_
                })
                st.dataframe(coef_df, hide_index=True)

# Таблица текущих сигналов
st.subheader("📋 Текущие сигналы")
table_data = []
for name in colors.keys():
    if st.session_state.predictions[name]:
        last = st.session_state.predictions[name][-1]
        table_data.append({
            'TF': name, 'Time': last['time'], 'Δgauss(raw)': f"{last['pred']:.6f}",
            'Norm': f"{last['pred']:.6f}", 'Dir': '⬆️ UP' if last['pred'] > 0 else '⬇️ DOWN'
        })

if st.session_state.meta_history:
    last_meta = st.session_state.meta_history[-1]
    table_data.append({
        'TF': 'META', 'Time': last_meta['time'], 'Δgauss(raw)': '-',
        'Norm': f"{last_meta['pred']:.6f}", 'Dir': '⬆️ UP' if last_meta['pred'] > 0 else '⬇️ DOWN'
    })

if table_data:
    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

# Статус
st.caption(f"Шаг симуляции: {current_step} | Источник: {'Синтетика' if is_synthetic else 'MT5'} | Буфер мета-модели: {len(st.session_state.meta_buffer)}")