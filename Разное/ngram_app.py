"""
N-gram Hunter v3 — Streamlit UI
Imports all logic from ngram_engine.py
Run with: streamlit run ngram_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json

import MetaTrader5 as mt5
from ngram_engine import (
    load_mt5, enrich_candles, build_level_database, compute_level_context,
    run_optuna, select_diverse, evaluate_ensemble, analyze_feature_importance,
    export_strategies, composite_score
)

st.set_page_config(page_title="N-gram Hunter v3 (Levels + Regime + Lifecycle)", layout="wide")

# ════════════════════════════════════════════
# UI
# ════════════════════════════════════════════
st.title("🕯️📐🌊 Candle N-gram Hunter v3 — Levels + Regime + Lifecycle")

st.sidebar.header("🎯 Instrument")
SYMBOLS = {
    "EURUSD H1": ("EURUSD", mt5.TIMEFRAME_H1),
    "BITCOIN H1": ("BITCOIN", mt5.TIMEFRAME_H1),
    "EURUSD M30": ("EURUSD", mt5.TIMEFRAME_M30),
    "EURUSD H4": ("EURUSD", mt5.TIMEFRAME_H4),
    "ETHEREUM H1": ("ETHEREUM", mt5.TIMEFRAME_H1),
}
choice = st.sidebar.selectbox("Instrument", list(SYMBOLS.keys()), index=0)
sym, tf_mt5 = SYMBOLS[choice]
n_bars_load = st.sidebar.slider("Bars", 3000, 30000, 15000, 1000)

st.sidebar.markdown("---")
st.sidebar.header("📐 Levels detection")
use_swings = st.sidebar.checkbox("Swing highs/lows", True)
use_volume = st.sidebar.checkbox("Volume nodes", True)
use_round = st.sidebar.checkbox("Round numbers", True)
swing_min_bars = st.sidebar.slider("Swing strictness (bars)", 2, 8, 3)
volume_top_n = st.sidebar.slider("Volume nodes to keep", 5, 30, 12)
round_step_pips = st.sidebar.slider("Round step (pips)", 25, 200, 50, 25)
cluster_tolerance = st.sidebar.slider("Level clustering (pips)", 3, 20, 8)

st.sidebar.markdown("---")
st.sidebar.header("🌊 Market Regimes (default)")
allow_trending_ui = st.sidebar.checkbox("Trending", True)
allow_ranging_ui = st.sidebar.checkbox("Ranging", True)
allow_breaking_ui = st.sidebar.checkbox("Breaking", True)
allow_choppy_ui = st.sidebar.checkbox("Choppy", False)

st.sidebar.markdown("---")
st.sidebar.header("⏳ Level Lifecycle")
max_exhaustion_ui = st.sidebar.slider("Max exhaustion", 0.3, 1.0, 0.8, 0.1)
min_freshness_ui = st.sidebar.slider("Min freshness", 0.2, 1.0, 0.4, 0.1)
min_reaction_ui = st.sidebar.slider("Min reaction quality", 0.0, 0.8, 0.2, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("🧠 Optuna")
n_trials = st.sidebar.slider("Trials", 100, 3000, 500, 50)
top_n = st.sidebar.slider("Top-N ensemble", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.header("💰 OOS test")
oos_days = st.sidebar.slider("OOS days", 20, 90, 30)
risk_pct = st.sidebar.slider("Risk %", 0.3, 2.0, 1.0, 0.1)
initial_deposit = st.sidebar.number_input("Deposit $", 1000, 100000, 10000, 1000)

run_opt = st.sidebar.button("🚀 START OPTUNA")
if st.sidebar.button("🔄 RESET"):
    for k in ["all_results", "top_strategies", "levels", "df", "df_train", "df_oos"]:
        st.session_state.pop(k, None)

# ════════════════════════════════════════════
# Load & Prepare Data
# ════════════════════════════════════════════
with st.spinner(f"Загружаем {choice}..."):
    raw = load_mt5(sym, tf_mt5, n_bars_load)
    df = enrich_candles(raw).dropna()

pip_value = 0.01 if "BITCOIN" in sym or "ETH" in sym else 0.0001

with st.spinner("Ищем уровни..."):
    levels = build_level_database(
        df, use_swings=use_swings, use_volume=use_volume, use_round=use_round,
        swing_min_bars=swing_min_bars, volume_top_n=volume_top_n,
        round_step_pips=round_step_pips, pip_value=pip_value,
        cluster_tolerance_pips=cluster_tolerance,
    )

with st.spinner("Считаем контекст уровней + lifecycle..."):
    df = compute_level_context(df, levels, pip_value=pip_value)

days = (df.index[-1] - df.index[0]).days
regime_counts = df["regime"].value_counts().to_dict()
regime_pcts = {k: v/len(df)*100 for k, v in regime_counts.items()}

st.info(
    f"**{choice}** • {len(df)} баров • {days} дней • "
    f"**{len(levels)} уровней** (avg strength: {np.mean([l['total_strength'] for l in levels]):.1f}) • "
    f"Regimes: T{regime_pcts.get(0,0):.0f}% / R{regime_pcts.get(1,0):.0f}% / B{regime_pcts.get(2,0):.0f}% / C{regime_pcts.get(3,0):.0f}%"
)

# ════════════════════════════════════════════
# Level Chart
# ════════════════════════════════════════════
with st.expander("📐 Найденные уровни на графике"):
    display_n = st.slider("Bars to show", 100, 1000, 400, 50, key="lvl_d")
    df_disp = df.tail(display_n)
    fig_lvl = go.Figure()
    fig_lvl.add_trace(go.Candlestick(
        x=df_disp.index, open=df_disp["open"], high=df_disp["high"],
        low=df_disp["low"], close=df_disp["close"], name=choice,
        opacity=0.8, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    max_strength = max((l["total_strength"] for l in levels), default=1)
    for lvl in levels:
        if not (df_disp["low"].min()*0.995 <= lvl["price"] <= df_disp["high"].max()*1.005): continue
        opacity = min(0.3 + lvl["total_strength"]/max_strength*0.6, 0.9)
        color = {"resistance":f"rgba(239,83,80,{opacity})","support":f"rgba(38,166,154,{opacity})",
                 "level":f"rgba(255,215,0,{opacity})","volume_node":f"rgba(179,136,255,{opacity})"}.get(lvl["type"],f"rgba(200,200,200,{opacity})")
        fig_lvl.add_hline(y=lvl["price"], line_dash="dash", line_color=color, line_width=1,
                         annotation_text=f"{lvl['type'][:3]}({lvl['total_strength']})", annotation_position="right")
    fig_lvl.update_layout(height=550, plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                         font=dict(color="#d4d4d4"), showlegend=False, yaxis=dict(gridcolor="#333"))
    fig_lvl.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])] if "EUR" in choice else [], rangeslider_visible=False)
    st.plotly_chart(fig_lvl, use_container_width=True)

with st.expander("📋 Top-20 уровней"):
    lvl_df = pd.DataFrame(levels).sort_values("total_strength", ascending=False).head(20)
    lvl_df["price"] = lvl_df["price"].round(5)
    lvl_df["sources"] = lvl_df["sources"].apply(lambda x: ", ".join(x))
    st.dataframe(lvl_df[["price","type","total_strength","touches","sources"]], use_container_width=True)

# ════════════════════════════════════════════
# Regime Chart
# ════════════════════════════════════════════
with st.expander("🌊 Режимы рынка на графике"):
    display_n2 = st.slider("Bars", 100, 1000, 500, 50, key="reg_d")
    df_reg = df.tail(display_n2)
    regime_colors = {0:"#2196F3",1:"#4CAF50",2:"#FF9800",3:"#9E9E9E"}
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Candlestick(x=df_reg.index, open=df_reg["open"], high=df_reg["high"],
                                     low=df_reg["low"], close=df_reg["close"], name=choice,
                                     opacity=0.6, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    for r, rn in [(0,"Trending"),(1,"Ranging"),(2,"Breaking"),(3,"Choppy")]:
        mask = df_reg["regime"] == r
        if mask.any():
            fig_reg.add_trace(go.Scatter(x=df_reg.index[mask], y=df_reg["high"][mask]*1.002,
                                        mode="markers", marker=dict(size=6,color=regime_colors[r],symbol="line-ew-open"),
                                        name=f"{rn}({mask.sum()})"))
    fig_reg.update_layout(height=500, plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                         font=dict(color="#d4d4d4"), showlegend=True, yaxis=dict(gridcolor="#333"),
                         legend=dict(orientation="h", y=-0.15))
    fig_reg.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])] if "EUR" in choice else [])
    st.plotly_chart(fig_reg, use_container_width=True)

    reg_stats = df.groupby("regime").agg(bars=("close","count"), avg_atr=("atr","mean")).round(4)
    reg_stats["%"] = (reg_stats["bars"]/len(df)*100).round(1)
    reg_stats.index = {0:"Trending",1:"Ranging",2:"Breaking",3:"Choppy"}.get
    reg_stats.index = pd.Index([{0:"Trending",1:"Ranging",2:"Breaking",3:"Choppy"}.get(i,"?") for i in reg_stats.index])
    st.dataframe(reg_stats, use_container_width=True)

# ════════════════════════════════════════════
# OPTUNA
# ════════════════════════════════════════════
if run_opt:
    st.header("🔬 Optuna: Levels + Regime + Lifecycle")

    if tf_mt5 == mt5.TIMEFRAME_H1: oos_bars = oos_days * 24
    elif tf_mt5 == mt5.TIMEFRAME_H4: oos_bars = oos_days * 6
    elif tf_mt5 == mt5.TIMEFRAME_M30: oos_bars = oos_days * 48
    else: oos_bars = oos_days * 24
    oos_bars = min(oos_bars, len(df) // 3)

    df_train = df.iloc[:-oos_bars]
    df_oos = df.iloc[-oos_bars:]
    st.caption(f"Train: {len(df_train)} bars • OOS: {len(df_oos)} bars ({oos_days}d)")

    progress = st.progress(0)
    status = st.empty()
    chart = st.empty()
    diag = st.empty()
    scores_history = []

    def _cb(study, trial):
        v = trial.value if trial.value is not None else -100
        scores_history.append(v)
        i = len(scores_history); best = max(scores_history)
        progress.progress(min(1.0, i/n_trials))
        status.write(f"Trial {i}/{n_trials} • best={best:.2f}")
        if i % 25 == 0 or i == n_trials:
            fig_p = go.Figure()
            fig_p.add_scatter(y=scores_history, mode="markers", marker=dict(size=3,opacity=0.4), name="trial")
            fig_p.add_scatter(y=pd.Series(scores_history).cummax(), mode="lines", line=dict(color="lime"), name="best")
            fig_p.update_layout(height=250, plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                               font=dict(color="#d4d4d4"), xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"))
            chart.plotly_chart(fig_p, use_container_width=True)

    with st.spinner(f"Running {n_trials} Optuna trials..."):
        all_results, study, fail_stats = run_optuna(df_train, n_trials=n_trials, callbacks=[_cb])

    st.session_state["all_results"] = all_results
    st.session_state["df_train"] = df_train
    st.session_state["df_oos"] = df_oos

    if not all_results:
        st.error(f"❌ Стратегий не найдено. fail_stats={fail_stats}\n\nПопробуй: включи allow_choppy, уменьши min_freshness, увеличь max_exhaustion")
    else:
        st.success(f"✅ {len(all_results)} стратегий найдено")

# ════════════════════════════════════════════
# ENSEMBLE + OOS
# ════════════════════════════════════════════
if "all_results" in st.session_state and st.session_state["all_results"]:
    all_results = st.session_state["all_results"]
    df_train = st.session_state["df_train"]
    df_oos = st.session_state["df_oos"]

    st.markdown("---")
    st.header(f"🏆 Top-{top_n} стратегий")
    top = select_diverse(all_results, top_n=top_n)

    rows = []
    for i, r in enumerate(top):
        p = r["params"]
        rows.append({
            "MAGIC": 2000+i, "K": p["K"], "hor": p["horizon"], "thr": round(p["threshold_pips"],1),
            "tp": round(p["tp_pips"],1), "sl": round(p["sl_pips"],1),
            "T": "✓" if p.get("allow_trending") else "-",
            "R": "✓" if p.get("allow_ranging") else "-",
            "B": "✓" if p.get("allow_breaking") else "-",
            "C": "✓" if p.get("allow_choppy") else "-",
            "fresh≥": round(p.get("min_freshness",0),1),
            "exh≤": round(p.get("max_exhaustion",1),1),
            "n": r["n_trades"], "wr%": round(r["wr"]*100,1),
            "pips": round(r["total_pips"],1), "sh": round(r["sharpe"],2), "pf": round(r["pf"],2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=380)

    # Regime breakdown лучшей
    with st.expander("🌊 Regime breakdown (лучшая)"):
        if top:
            rb = top[0].get("regime_breakdown", {})
            rb_data = {
                "Regime": ["Trending","Ranging","Breaking","Choppy"],
                "Trades": [rb.get("regime_0_trades",0),rb.get("regime_1_trades",0),rb.get("regime_2_trades",0),rb.get("regime_3_trades",0)],
                "WR%": [rb.get("regime_0_wr",0),rb.get("regime_1_wr",0),rb.get("regime_2_wr",0),rb.get("regime_3_wr",0)],
                "Pips": [rb.get("regime_0_pips",0),rb.get("regime_1_pips",0),rb.get("regime_2_pips",0),rb.get("regime_3_pips",0)],
            }
            st.dataframe(pd.DataFrame(rb_data), use_container_width=True)

    st.markdown("---")
    st.header(f"🎬 OOS Ensemble ({oos_days}д)")
    port, metrics, per_magic = evaluate_ensemble(df_train, df_oos, top, risk_pct=risk_pct, initial_deposit=initial_deposit)

    st.dataframe(pd.DataFrame(per_magic), use_container_width=True)

    if metrics:
        c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
        c1.metric("Trades", metrics["trades"])
        c2.metric("PnL p", f"{metrics['total_pips']:+.1f}")
        c3.metric("PnL $", f"{metrics['total_usd']:+.2f}")
        c4.metric("ROI%", f"{metrics['roi_pct']:+.2f}%")
        c5.metric("WR", f"{metrics['wr']*100:.1f}%")
        c6.metric("Sh", f"{metrics['sharpe']:.2f}")
        c7.metric("PF", f"{metrics['pf']:.2f}")

        c1,c2,c3 = st.columns(3)
        c1.metric("Max DD $", f"-${metrics['max_dd_usd']:,.2f}")
        c2.metric("MAGICs", metrics["magic_count"])

        # Equity chart
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=port["exit_time"], y=port["equity_usd"], mode="lines",
                                   line=dict(color="#00e676" if metrics["total_usd"]>=0 else "#ff5252", width=2.5),
                                   fill="tozeroy",
                                   fillcolor="rgba(0,230,118,0.15)" if metrics["total_usd"]>=0 else "rgba(255,82,82,0.15)"))
        fig_eq.add_hline(y=initial_deposit, line_dash="dot", line_color="gray", opacity=0.5)
        fig_eq.update_layout(height=400, plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                           font=dict(color="#d4d4d4"), yaxis=dict(gridcolor="#333"))
        st.plotly_chart(fig_eq, use_container_width=True)

        # Feature importance
        with st.expander("🧠 Feature importance"):
            imp_df, cat_pct = analyze_feature_importance(top[0])
            st.dataframe(imp_df, use_container_width=True, height=500)
            st.info(f"📐Levels:{cat_pct.get('Level',0)}% • 🌊Regime:{cat_pct.get('Regime',0)}% • ⏳Lifecycle:{cat_pct.get('Lifecycle',0)}% • 🕯️Candles:{cat_pct.get('Candle',0)}%")

        # Export
        export_json = export_strategies(top)
        st.download_button("💾 Export MT5 EA (JSON)", data=export_json,
                          file_name=f"ngram_v3_{choice.replace(' ','_')}.json", mime="application/json")
else:
    if not run_opt:
        st.markdown("""
        ## 🎯 V3: Свечи + Уровни + Режимы + Lifecycle
        
        **Новое в v3:**
        1. 🌊 **Market Regime** — ADX + BB + ATR → тренд/флэт/пробой/хай
        2. ⏳ **Level Lifecycle** — возраст, усталость, качество реакций
        3. 🧠 **Regime-aware Optuna** — сама выбирает режимы для торговли
        4. ⏱️ **Freshness filter** — торгуем на свежих, не на уставших уровнях
        
        **Запуск:** выбери инструмент → настрой фильтры → 🚀 START OPTUNA
        """)
