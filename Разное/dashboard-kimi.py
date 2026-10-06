
# Дополняем файл информацией о подключении MT5 и создаём финальную версию

final_code = '''"""
Мульти-Таймфреймный дашборд EURUSD (MT5 + LightGBM)
====================================================
Интерактивный дашборд для визуализации и пошагового тестирования 
LightGBM-моделей на разных таймфреймах (5m, 10m, 15m, 30m, 60m) 
по паре EURUSD с данными MetaTrader 5.

Запуск: streamlit run dashboard.py
Требуется: запущенный MT5 + библиотека MetaTrader5

Установка зависимостей:
    pip install streamlit pandas numpy scipy lightgbm plotly MetaTrader5
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── Попытка импорта MetaTrader5 ─────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ─── Конфигурация страницы ──────────────────────────────────────────────────
st.set_page_config(
    page_title="EURUSD MTF Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS для компактного отображения ────────────────────────────────────────
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    .stButton>button {width: 100%; font-weight: 600;}
    .stDataFrame {font-size: 12px;}
    div[data-testid="stMetricValue"] {font-size: 16px;}
    h1 {font-size: 22px !important; margin-bottom: 0.5rem !important;}
    h2 {font-size: 18px !important; margin-bottom: 0.3rem !important;}
    h3 {font-size: 15px !important; margin-bottom: 0.2rem !important;}
</style>
""", unsafe_allow_html=True)

# ─── Константы ──────────────────────────────────────────────────────────────
TFS = {
    '5m':  {'minutes': 5,   'color': '#1f77b4', 'train_min': 100, 'train_max': 1000, 'train_default': 300},
    '10m': {'minutes': 10,  'color': '#d62728', 'train_min': 50,  'train_max': 500,  'train_default': 150},
    '15m': {'minutes': 15,  'color': '#2ca02c', 'train_min': 50,  'train_max': 500,  'train_default': 150},
    '30m': {'minutes': 30,  'color': '#9467bd', 'train_min': 50,  'train_max': 500,  'train_default': 150},
    '60m': {'minutes': 60,  'color': '#8c564b', 'train_min': 50,  'train_max': 500,  'train_default': 150},
}

TF_ORDER = ['5m', '10m', '15m', '30m', '60m']
TEST_BARS = 100  # Последние 100 баров 5m откладываются под тест

# ─── Инициализация session_state ────────────────────────────────────────────
def init_session():
    defaults = {
        'data_loaded': False,
        'df_5m': None,
        'df_5m_full': None,
        'models': {},
        'predictions_history': {},
        'current_step': 0,
        'simulation_started': False,
        'last_train_params': {},
        'use_real_mt5': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─── Функции обработки данных ───────────────────────────────────────────────

def generate_synthetic_data(n_bars=15100):
    """
    Генерация синтетических данных EURUSD M5 для демонстрации.
    Структура данных идентична выходу mt5.copy_rates_from_pos().
    """
    np.random.seed(42)
    base_price = 1.0850
    
    # Генерация временных меток (только будни, торговые часы)
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    times = []
    current = start_time
    while len(times) < n_bars:
        if current.weekday() < 5:  # Пн-Пт
            if 6 <= current.hour < 22:  # Торговые часы ~06:00-22:00 UTC
                times.append(current)
        current += timedelta(minutes=5)
    
    times = times[:n_bars]
    n = len(times)
    
    # Реалистичная генерация цен
    returns = np.random.normal(0, 0.00015, n)
    trend = np.sin(np.linspace(0, 8*np.pi, n)) * 0.002
    trend += np.sin(np.linspace(0, 2*np.pi, n)) * 0.001
    
    vol_regime = np.ones(n)
    for i in range(0, n, 500):
        vol_regime[i:min(i+200, n)] *= np.random.choice([0.5, 1.0, 2.0])
    
    returns = returns * vol_regime + trend / n
    close = base_price + np.cumsum(returns)
    
    high = close + np.abs(np.random.normal(0, 0.0001, n))
    low = close - np.abs(np.random.normal(0, 0.0001, n))
    open_price = np.roll(close, 1)
    open_price[0] = base_price
    
    # Корректировка OHLC
    for i in range(n):
        o, h, l, c = open_price[i], high[i], low[i], close[i]
        h = max(h, o, c)
        l = min(l, o, c)
        high[i], low[i] = h, l
    
    df = pd.DataFrame({
        'time': times,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'tick_volume': np.random.randint(100, 5000, n),
        'spread': np.random.randint(1, 5, n),
        'real_volume': np.random.randint(1000, 50000, n),
    })
    
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df = df[df.index.dayofweek < 5]  # Фильтр: только будни
    
    return df


def load_mt5_data(symbol="EURUSD", timeframe=None, count=15000):
    """
    Загрузка данных из MetaTrader 5.
    
    Использует mt5.copy_rates_from_pos(symbol, timeframe, 0, count).
    Требует запущенного терминала MT5.
    """
    if not MT5_AVAILABLE:
        raise ImportError("Библиотека MetaTrader5 не установлена. Установите: pip install MetaTrader5")
    
    if not mt5.initialize():
        raise ConnectionError(f"Не удалось подключиться к MT5: {mt5.last_error()}")
    
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_M5
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    if rates is None or len(rates) == 0:
        raise ValueError(f"Не удалось получить данные для {symbol}. Убедитесь, что символ доступен в Market Watch.")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[df.index.dayofweek < 5]  # Фильтр: только будни
    
    mt5.shutdown()
    return df


def resample_ohlc(df_5m, minutes):
    """Ресэмплинг M5 → старший таймфрейм (OHLC: first/max/min/last)."""
    rule = f'{minutes}T'
    resampled = df_5m.resample(rule, label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'tick_volume': 'sum',
        'spread': 'last',
        'real_volume': 'sum',
    }).dropna()
    return resampled


def add_gaussian_features(df, sigma=5):
    """Добавление гауссовой конволюции и её дельты."""
    df = df.copy()
    df['gauss'] = gaussian_filter1d(df['close'].values, sigma=sigma)
    df['delta_gauss'] = df['gauss'].diff()
    df['target'] = df['delta_gauss'].shift(-1)  # Целевая: изменение на 1 бар вперёд
    return df


def build_features(df, lags=10):
    """Построение лаговых признаков (только прошлое, shift(1))."""
    df = df.copy()
    
    # Гауссовы лаги gc_lag_1..10
    for i in range(1, lags + 1):
        df[f'gc_lag_{i}'] = df['gauss'].shift(i)
    
    # OHLC разностные лаги open_dif_1..10, high_dif_1..10 и т.д.
    for col in ['open', 'high', 'low', 'close']:
        df[f'{col}_dif'] = df[col].diff()
        for i in range(1, lags + 1):
            df[f'{col}_dif_lag_{i}'] = df[f'{col}_dif'].shift(i)
    
    return df


def prepare_tf_data(df_5m, tf_name, train_bars, lags=10, sigma=5):
    """Подготовка данных для одного таймфрейма."""
    tf_info = TFS[tf_name]
    
    if tf_name == '5m':
        df = df_5m.copy()
    else:
        df = resample_ohlc(df_5m, tf_info['minutes'])
    
    # Гаусс + признаки
    df = add_gaussian_features(df, sigma=sigma)
    df = build_features(df, lags=lags)
    
    # MA20
    df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
    
    # Отбор признаков
    feature_cols = []
    for i in range(1, lags + 1):
        feature_cols.append(f'gc_lag_{i}')
    for col in ['open', 'high', 'low', 'close']:
        for i in range(1, lags + 1):
            feature_cols.append(f'{col}_dif_lag_{i}')
    
    # Убираем NaN
    df = df.dropna(subset=['target'] + feature_cols)
    
    return df, feature_cols


def train_model(df, feature_cols, tf_name):
    """Обучение LightGBM Regressor."""
    X = df[feature_cols].values
    y = df['target'].values
    
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def predict_for_bar(df, model, feature_cols):
    """Предсказание delta_gauss для последнего бара."""
    last_row = df[feature_cols].iloc[-1:].values
    pred = model.predict(last_row)[0]
    return pred


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор источника данных
    st.subheader("Источник данных")
    use_mt5 = st.toggle(
        "Использовать реальный MT5", 
        value=st.session_state.use_real_mt5,
        disabled=not MT5_AVAILABLE,
        help="Требуется запущенный терминал MetaTrader 5"
    )
    st.session_state.use_real_mt5 = use_mt5
    
    if not MT5_AVAILABLE:
        st.info("ℹ️ MetaTrader5 не установлен. Используется синтетические данные.")
    
    st.markdown("---")
    
    st.subheader("Обучающие окна")
    train_sizes = {}
    for tf in TF_ORDER:
        info = TFS[tf]
        train_sizes[tf] = st.slider(
            f"{tf} — бары обучения",
            min_value=info['train_min'],
            max_value=info['train_max'],
            value=info['train_default'],
            step=10,
            key=f"train_{tf}"
        )
    
    st.subheader("Параметры")
    sigma = st.slider("Sigma гаусса", 1, 20, 5, key="sigma")
    lags = st.slider("Кол-во лагов", 1, 20, 10, key="lags")
    
    st.markdown("---")
    
    if st.button("🔄 Загрузить данные", use_container_width=True, type="primary"):
        with st.spinner("Загрузка данных..."):
            try:
                if use_mt5 and MT5_AVAILABLE:
                    df_5m = load_mt5_data(symbol="EURUSD", count=15000)
                    source = "MetaTrader 5"
                else:
                    df_5m = generate_synthetic_data(n_bars=15100)
                    source = "Синтетические данные"
                
                st.session_state.df_5m_full = df_5m.copy()
                st.session_state.df_5m = df_5m.iloc[:-TEST_BARS].copy()
                st.session_state.data_loaded = True
                st.session_state.current_step = 0
                st.session_state.simulation_started = False
                st.session_state.predictions_history = {tf: [] for tf in TF_ORDER}
                st.session_state.models = {}
                
                st.success(f"✅ Загружено {len(df_5m)} баров M5 (+{TEST_BARS} тестовых) из {source}")
                
            except Exception as e:
                st.error(f"❌ Ошибка загрузки: {str(e)}")
    
    if st.session_state.data_loaded:
        st.markdown(f"**Тренировочных баров:** {len(st.session_state.df_5m)}")
        st.markdown(f"**Тестовых баров:** {TEST_BARS}")
        st.markdown(f"**Период:** {st.session_state.df_5m.index[0].strftime('%Y-%m-%d')} → {st.session_state.df_5m.index[-1].strftime('%Y-%m-%d')}")


# ─── Основной контент ─────────────────────────────────────────────────────────
st.title("📊 Мульти-Таймфреймный дашборд EURUSD")
st.caption("MT5 + LightGBM | 5m · 10m · 15m · 30m · 60m")

if not st.session_state.data_loaded:
    st.info("👈 Нажмите **'Загрузить данные'** в сайдбаре для начала работы.")
    
    # Демонстрационный график-заглушка
    fig_demo = go.Figure()
    fig_demo.add_trace(go.Scatter(
        x=[1, 2, 3], y=[1, 2, 1],
        mode='lines', name='Демо',
        line=dict(color='gray', dash='dash')
    ))
    fig_demo.update_layout(
        title="Ожидание загрузки данных...",
        height=400,
        template='plotly_dark'
    )
    st.plotly_chart(fig_demo, use_container_width=True)
    
    st.stop()

# ─── Кнопки управления симуляцией ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

with col1:
    if st.button("▶️ Старт симуляции", use_container_width=True, 
                 disabled=st.session_state.simulation_started,
                 type="primary"):
        st.session_state.simulation_started = True
        st.session_state.current_step = 0
        st.session_state.predictions_history = {tf: [] for tf in TF_ORDER}
        st.rerun()

with col2:
    if st.button("➡️ Следующий бар", use_container_width=True,
                 disabled=not st.session_state.simulation_started or st.session_state.current_step >= TEST_BARS):
        st.session_state.current_step += 1
        st.rerun()

with col3:
    if st.button("🔁 Сброс", use_container_width=True):
        st.session_state.current_step = 0
        st.session_state.simulation_started = False
        st.session_state.predictions_history = {tf: [] for tf in TF_ORDER}
        st.session_state.models = {}
        st.rerun()

with col4:
    st.markdown(f"**Шаг:** `{st.session_state.current_step} / {TEST_BARS}`")
    progress = st.session_state.current_step / TEST_BARS if TEST_BARS > 0 else 0
    st.progress(progress)

# ─── Подготовка данных для текущего шага ─────────────────────────────────────
df_5m_train = st.session_state.df_5m.copy()
df_5m_full = st.session_state.df_5m_full.copy()

# Добавляем тестовые бары по шагам
if st.session_state.simulation_started and st.session_state.current_step > 0:
    test_bars_to_add = df_5m_full.iloc[-TEST_BARS:-TEST_BARS + st.session_state.current_step]
    df_5m_current = pd.concat([df_5m_train, test_bars_to_add])
else:
    df_5m_current = df_5m_train.copy()

# ─── Обучение / Переобучение моделей ─────────────────────────────────────────
current_params = {
    'train_sizes': {tf: train_sizes[tf] for tf in TF_ORDER},
    'sigma': sigma,
    'lags': lags,
    'step': st.session_state.current_step,
}

need_retrain = (
    not st.session_state.models or
    current_params != st.session_state.last_train_params
)

if st.session_state.simulation_started and need_retrain:
    with st.spinner("Обучение моделей..."):
        st.session_state.models = {}
        st.session_state.feature_cols = {}
        
        for tf in TF_ORDER:
            df_tf, feature_cols = prepare_tf_data(
                df_5m_current, tf, 
                train_sizes[tf], 
                lags=lags, sigma=sigma
            )
            
            # Берём последние N баров для обучения
            train_df = df_tf.tail(train_sizes[tf])
            
            if len(train_df) > 20:
                model = train_model(train_df, feature_cols, tf)
                st.session_state.models[tf] = model
                st.session_state.feature_cols[tf] = feature_cols
            else:
                st.warning(f"Недостаточно данных для {tf}")
        
        st.session_state.last_train_params = current_params.copy()

# ─── Предсказания для текущего шага ──────────────────────────────────────────
if st.session_state.simulation_started and st.session_state.models:
    for tf in TF_ORDER:
        if tf not in st.session_state.models:
            continue
            
        df_tf, _ = prepare_tf_data(
            df_5m_current, tf, 
            train_sizes[tf], 
            lags=lags, sigma=sigma
        )
        
        if len(df_tf) > 0:
            pred = predict_for_bar(
                df_tf, 
                st.session_state.models[tf], 
                st.session_state.feature_cols[tf]
            )
            
            last_time = df_tf.index[-1]
            
            # Накопление истории предсказаний
            hist = st.session_state.predictions_history[tf]
            if len(hist) < st.session_state.current_step:
                hist.append({'time': last_time, 'pred': pred, 'step': st.session_state.current_step})
                st.session_state.predictions_history[tf] = hist

# ─── Визуализация ───────────────────────────────────────────────────────────
df_plot = df_5m_current.copy()
df_plot['ma20'] = df_plot['close'].rolling(window=20, min_periods=1).mean()
df_plot['gauss'] = gaussian_filter1d(df_plot['close'].values, sigma=sigma)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.7, 0.3],
    specs=[[{"type": "candlestick"}], [{"type": "scatter"}]]
)

# ─── Подграфик 1: Свечи + MA20 (70%) ───────────────────────────────────────
fig.add_trace(
    go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='EURUSD',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        opacity=0.7,
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_plot.index,
        y=df_plot['ma20'],
        mode='lines',
        name='MA20',
        line=dict(color='#ff9800', width=1.5),
        opacity=0.9,
    ),
    row=1, col=1
)

# ─── Подграфик 2: Предсказания delta_gauss (30%) ──────────────────────────
fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)

for tf in TF_ORDER:
    hist = st.session_state.predictions_history[tf]
    if len(hist) > 0:
        times = [h['time'] for h in hist]
        preds = [h['pred'] for h in hist]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=preds,
                mode='lines+markers',
                name=f'{tf}',
                line=dict(color=TFS[tf]['color'], width=2),
                marker=dict(size=6, symbol='circle'),
                legendgroup=tf,
            ),
            row=2, col=1
        )

# ─── Разметка тестовой зоны ─────────────────────────────────────────────────
if st.session_state.simulation_started and st.session_state.current_step > 0:
    test_start = df_5m_train.index[-1]
    test_end = df_plot.index[-1]
    
    fig.add_vrect(
        x0=test_start, x1=test_end,
        fillcolor="yellow", opacity=0.05,
        layer="below", line_width=0,
        row=1, col=1
    )
    
    # Вертикальная линия текущего шага
    if len(df_plot) > len(df_5m_train):
        current_time = df_plot.index[len(df_5m_train) + st.session_state.current_step - 1]
        fig.add_vline(x=current_time, line_dash="dash", line_color="white", line_width=1, opacity=0.5)

# ─── Layout ─────────────────────────────────────────────────────────────────
fig.update_layout(
    template='plotly_dark',
    height=700,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor='rgba(0,0,0,0.3)',
        font=dict(size=11)
    ),
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    paper_bgcolor='#0e1117',
    plot_bgcolor='#0e1117',
    font=dict(color='#e0e0e0', size=11),
)

# Rangebreaks: скрыты выходные (sat-mon) + ночные перерывы
fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[22, 6], pattern="hour"),
    ],
    row=1, col=1
)
fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[22, 6], pattern="hour"),
    ],
    row=2, col=1
)

fig.update_yaxes(title_text="Цена", row=1, col=1)
fig.update_yaxes(title_text="ΔGauss", row=2, col=1)

st.plotly_chart(fig, use_container_width=True, key="main_chart")

# ─── Таблица текущих сигналов ───────────────────────────────────────────────
st.subheader("📋 Текущие сигналы delta_gauss")

signal_data = []
for tf in TF_ORDER:
    hist = st.session_state.predictions_history[tf]
    if len(hist) > 0:
        last_pred = hist[-1]['pred']
        signal_data.append({
            'TF': tf,
            'Предсказание': f"{last_pred:.6f}",
            'Направление': '🟢 LONG' if last_pred > 0 else '🔴 SHORT' if last_pred < 0 else '⚪ NEUTRAL',
            'Шаг': hist[-1]['step'],
        })
    else:
        signal_data.append({
            'TF': tf,
            'Предсказание': '—',
            'Направление': '⏳ Ожидание',
            'Шаг': '—',
        })

signal_df = pd.DataFrame(signal_data)
st.dataframe(
    signal_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        'TF': st.column_config.TextColumn('Таймфрейм', width='small'),
        'Предсказание': st.column_config.TextColumn('ΔGauss', width='medium'),
        'Направление': st.column_config.TextColumn('Сигнал', width='medium'),
        'Шаг': st.column_config.NumberColumn('Шаг', width='small'),
    }
)

# ─── Метрики ──────────────────────────────────────────────────────────────────
if st.session_state.simulation_started and len(st.session_state.predictions_history['5m']) > 0:
    st.subheader("📈 Статистика предсказаний")
    
    cols = st.columns(5)
    for i, tf in enumerate(TF_ORDER):
        hist = st.session_state.predictions_history[tf]
        if len(hist) > 0:
            preds = [h['pred'] for h in hist]
            with cols[i]:
                st.metric(
                    label=f"{tf} mean ΔGauss",
                    value=f"{np.mean(preds):.6f}",
                    delta=f"σ={np.std(preds):.6f}"
                )

# ─── Информация о моделях ───────────────────────────────────────────────────
with st.expander("🔧 Информация о моделях"):
    for tf in TF_ORDER:
        if tf in st.session_state.models:
            model = st.session_state.models[tf]
            st.markdown(f"**{tf}:** n_estimators={model.n_estimators}, max_depth={model.max_depth}, lr={model.learning_rate}")
        else:
            st.markdown(f"**{tf}:** Модель не обучена")
    
    st.markdown("---")
    st.markdown(f"**Признаки:** {lags} лагов × (gc + open_dif + high_dif + low_dif + close_dif) = {lags * 5} признаков")
    st.markdown(f"**Гаусс:** sigma={sigma}")
    st.markdown(f"**Целевая переменная:** delta_gauss.shift(-1)")
    st.markdown(f"**Фильтр:** только будни (dayofweek < 5)")
    st.markdown(f"**Ресэмплинг:** M5 → 10m/15m/30m/60m (first/max/min/last)")
'''

with open('/mnt/agents/output/dashboard.py', 'w', encoding='utf-8') as f:
    f.write(final_code)

print("✅ Финальная версия dashboard.py создана!")
print(f"📁 Размер файла: {len(final_code)} символов")
