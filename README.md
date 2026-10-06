# METATRADER

Лаборатория торговых стратегий для MetaTrader 5, разрабатываемых совместно с ИИ-ассистентами (DeepSeek, Gemini, QWEN, Kimi). В репозитории — код всех версий и экспериментов, обученные модели, результаты walk-forward и бэктестов, сотни equity-графиков, дашборды и техническое задание.

## 🎯 Суть системы на словах

Проект — итеративная «песочница»: одна и та же идея стратегии переписывается версиями (v1 → v5), проверяется оптимизацией Optuna, walk-forward и построчным «живым» симулятором, а затем сводится в ансамбль. Основные линии:

1. **RSI-ансамбль.** `rsi_optimize.py` — оптимизация параметров RSI-стратегии на Optuna по той же методике, что и MA-чемпионы. `rsi_simulator.py` → `rsi_simulator_v4.py` — бар-по-барный живой симулятор (прогон месяца за ~2 минуты): исправлен вход на открытии следующего бара, кулдаун после SL, ATR по Уайлдеру, закрытие по цене своего ТФ. `rsi_live_session.py` — live-сборка на 15 чемпионов (M15+M30+H1) с сессионным фильтром 08:00–17:00; `rsi_hourly_analysis.py` — поиск оптимальных торговых часов.

2. **MA-ансамбль** (`MA/`). Серия Optuna v1 → v9: перебор комбинаций скользящих, число чемпионов, длина train-окна; в v9 исправлен look-ahead bias (`o[i] → o[i+1]`). Walk-forward, мета-оптимизация, сравнение MA vs RSI, heatmaps/grid-таблицы, champions-JSON по ТФ, живая сессия и экспорт стратегии в MQL5 (`EURUSD_MA_Ensemble.mq5`).

3. **CatBoost/LightGBM-ансамбли.** `train_v5*.py` / `bot_v5*.py` / `config_v5.py` — обучалка и живой бот на комплексных каузальных признаках (только прошлое); `ЛУЧШИЙ1_build_ensemble_v5.py-Обучалка-Сборщик.py` и `ЛУЧШИЙ1_trade_bot_v5.py-Живой Бот—считывает-конфиг-V5.py` — финальная сборка: обучалка собирает ансамбль, бот читает конфиг и считает те же фичи. `ensemble_catboost*.py`, `ensemble_backtest.py` — агрегация стратегий с фильтрацией и фиксированным агрегатом; кэш моделей `models_cache/*.pkl`.

4. **Анализатор → таблица → БОТ.** `АНАЛИЗАТОР-в-ТАБЛИЦУ*.py` — сканер признаков (BTC и фоrex), `Анализ-и-БОТ-при-ответе-Yes*.py` — цепочка «анализ → подтверждение Yes → сделка» с единой функцией расчёта каузальных фичей, синхронизированной между сканером и ботом (версии с фиксами брокера/DTF).

5. **Исследовательские ветки.** `analysis_v1 → analysis_v5-2-эквити-ДипСик-OK-*` — эволюция анализа с ИИ: производные, учёт lookahead, equity по всем парам и таймфреймам суммарно, версии «разница» и «БОТ35»; `eurusd_*.py` — отдельная линия по EURUSD (ATR-стоп, RR, hold-периоды, фильтры); `ver_1–ver_3.py` — минимальные live-боты со SL; `equity_test*.py`, `experiment_grid.py`, `optimize_n*.py` — проверка логики equity и перебор N.

6. **Ансамбли с ИИ.** `Сборка-ансамбля-Джемини*.py`, `обучалка-джемини.py`, `Сборка-ансамбля-Джемини-сам-БОТ*.py` — сборка ансамбля моделей (в т.ч. скользящее окно 20 баров) и самобот; `ver`-файлы DEEPSEEK/QWEN — параллельные доработки тех же идей.

7. **Разное** (`Разное/`): Bollinger Bands (`bb_optimize.py`), свечные паттерны (`candl*.py`), `ngram_app.py`, дашборды `dashboard*.py` и `dashboard-kimi.py`, оптимизация BTC (`opt_BTC_60m.db`), материалы и ТЗ (PDF/DOCX).

Результаты: `backtest_results/` — 248 equity-графиков (пары × таймфреймы × Hold1–10); `walk_forward_results/` — walk-forward ансамбля; `ensemble_results(_v2|_v3|_v4)/` — equity топ-35 стратегий; `models_cache/` — LightGBM-модели по таймфреймам.

## 📈 Графики эквити

| RSI: тест | RSI: симулятор | RSI: часы |
|---|---|---|
| ![rsi test](rsi_test.png) | ![rsi sim](rsi_sim_equity.png) | ![rsi hourly](rsi_hourly.png) |

| MA: equity | Ver1 check | Ver2 check |
|---|---|---|
| ![ma equity](eur_ma_equity.png) | ![ver1](ver_1_check.png) | ![ver2](ver_2_check.png) |

Ансамбли (v1 → v4) и walk-forward:

| Ensemble v1 | v2 | v3 | v4 |
|---|---|---|---|
| ![e1](ensemble_results/ensemble_equity.png) | ![e2](ensemble_results_v2/ensemble_v2_equity.png) | ![e3](ensemble_results_v3/ensemble_v3_equity.png) | ![e4](ensemble_results_v4/ensemble_v4_equity.png) |

| Walk-forward | Optuna | Со SL |
|---|---|---|
| ![wf](walk_forward_results/walk_forward_ensemble.png) | ![wfo](walk_forward_results/walk_forward_ensemble_optuna.png) | ![wfsl](walk_forward_results/walk_forward_ensemble_with_sl.png) |

MA-оптимизация (`MA/`):

| v5 walk-forward | v9 итоговый | v8 heatmap |
|---|---|---|
| ![ma5](MA/ma_v5_walkforward.png) | ![ma9](MA/ma_v9_final_test.png) | ![ma8](MA/ma_v8_heatmaps.png) |

| MA vs RSI | MA session compare |
|---|---|
| ![mavsr](MA/ma_vs_rsi.png) | ![masess](MA/ma_session_compare.png) |

Примеры из `backtest_results/` (пары × ТФ × Hold):

| EURUSD H4 Hold3 | GBPUSD M15 Hold2 | EURUSD M5 Hold5 |
|---|---|---|
| ![h3](backtest_results/EURUSD_H4_Hold3.png) | ![h2](backtest_results/GBPUSD_M15_Hold2.png) | ![h5](backtest_results/EURUSD_M5_Hold5.png) |

| AUDUSD H1 Hold1 | USDJPY H4 Hold4 | NZDUSD M15 Hold7 |
|---|---|---|
| ![a1](backtest_results/AUDUSD_H1_Hold1.png) | ![u4](backtest_results/USDJPY_H4_Hold4.png) | ![n7](backtest_results/NZDUSD_M15_Hold7.png) |

## 📁 Структура папок

| Путь | Назначение |
|------|------------|
| `rsi_*.py`, `rsi_*.txt` | RSI-ансамбль: оптимизация, симулятор v1–v4, live-сессия |
| `MA/` | MA-ансамбль Optuna v1–v9, MQL5-экспорт, champions |
| `analysis_*.py`, `eurusd_*.py`, `ver_*.py`, `equity_test*.py` | исследовательские ветки анализа и минимальные боты |
| `train_v5*.py`, `bot_v5*.py`, `config_v5.py`, `ensemble_*.py`, `ЛУЧШИЙ1_*` | обучалка + живой бот + ансамбли CatBoost/LightGBM |
| `АНАЛИЗАТОР-*`, `Анализ-и-БОТ-*`, `Arena-*.py` | сканер → таблица → бот (BTC/forex) |
| `Сборка-ансамбля-Джемини*.py`, `обучалка-джемини.py` | ансамбли с ИИ, самобот |
| `METATRADER5/` | версии «АНАЛИЗАТОР-в-ТАБЛИЦУ» под MT5 |
| `Разное/` | Bollinger, свечные паттерны, ngram, дашборды, PDF/DOCX, BTC-opt |
| `копии версий/` | архив копий версий скриптов |
| `backtest_results/`, `walk_forward_results/`, `ensemble_results*/`, `models_cache/` | графики, walk-forward, модели, топ-35 стратегий |
| `*.txt` | логи запусков рядом с каждым скриптом |
| `ПолноеТЗ.docx`, `ТЗ от Джем.docx`, `Разное/TZ4.pdf` | техническое задание |

## 🚀 Запуск

```bash
pip install MetaTrader5 pandas numpy lightgbm catboost optuna matplotlib scipy
python rsi_simulator_v4.py            # живой симулятор RSI (~2 мин на месяц)
python MA/ma_optuna_v9_corrected.py   # MA-ансамбль с исправленным lookahead
python ЛУЧШИЙ1_trade_bot_v5.py-Живой Бот—считывает-конфиг-V5.py   # живой бот V5
```

Требуется запущенный терминал MetaTrader 5 и доступные исторические данные по инструментам. Каждый `.py` сопровождается `.txt` — логом последнего прогона.

## ⬆️ Что не выгружено в репозиторий

Через `.gitignore` отсечены: `__pycache__/`, логи (`*.log`), служебный каталог `catboost_info/`, временные файлы Word (`~$*`). `.venv` в проекте нет.

---

> **ВНИМАНИЕ:** любые торговые стратегии — высокий риск потери капитала. Всё выложено исключительно в ознакомительных целях. Проверяйте на демо.
