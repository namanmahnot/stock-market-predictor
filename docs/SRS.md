# Software Requirements Specification

## Stock Market Price Prediction System

**Version 1.2 — October 2026**
AI Laboratory Project · Artificial Intelligence and Machine Learning
Prediction · Technical Analysis · Model Evaluation · Visualization

> Version 1.2 adds the NSE bulk-deal module (FR-15). Version 1.1 revised the proposal
> (v1.0) to describe the system **as built**. Requirements
> that were implemented are marked with the module and test that satisfy them; the two
> requirements that were deliberately not implemented are stated as such in §4 and §21
> rather than quietly dropped.

---

## 1. Introduction

### 1.1 Purpose

This project develops an AI-powered web application that analyses historical stock market
data and generates forecasts of future stock prices. It helps users explore price trends,
compare machine learning algorithms on identical data, understand historical market
behaviour, and evaluate forecast accuracy through an interactive dashboard.

The system is intended for educational and analytical purposes. It does not guarantee future
market prices and does not provide personalised investment advice.

### 1.2 Problem statement

Stock prices move in response to historical trends, trading activity, economic conditions and
unexpected events. Manually analysing large amounts of price data is difficult for beginners,
and a prediction system built around a single algorithm offers no way to tell whether that
algorithm is doing anything useful.

The harder problem is evaluation rather than prediction. A model that forecasts tomorrow's
closing price as "roughly today's closing price" will show a small average error and an
impressive-looking chart while containing no predictive information at all. A system that
reports only MAE, or only a chart of predicted-versus-actual prices, cannot distinguish that
case from a genuinely informative model.

This project therefore combines multiple machine learning approaches, technical indicators,
historical visualisation and a standardised evaluation protocol in which **every model is
measured against a naive baseline on data that was not available when it was fitted**.

### 1.3 Proposed solution

The system allows users to:

- search for a stock by ticker symbol;
- view historical prices, traded volume and technical indicators;
- forecast future prices over configurable horizons;
- compare several AI/ML algorithms on identical data, splits and targets;
- evaluate models using MAE, RMSE, MAPE, directional accuracy, R² and skill against a baseline;
- inspect which features drove a tree model's prediction;
- simulate a historical trading rule with transaction costs;
- save watchlists and prediction history;
- export analytical reports as CSV or PDF.

---

## 2. Project objectives

1. Develop a responsive web application for stock market analysis and price prediction. ✔
2. Collect and process historical stock market data automatically. ✔
3. Implement and compare multiple machine learning algorithms. ✔ (8 implemented, 6 in the default set)
4. Use time-series forecasting methods to predict future prices. ✔
5. Display predictions and historical trends using interactive visualisations. ✔
6. Evaluate forecast performance using appropriate statistical metrics. ✔
7. Provide an easy-to-use dashboard for students and general users. ✔
8. Store user preferences, watchlists, model results and prediction history. ✔
9. Follow proper machine learning practice and avoid misleading predictions. ✔ (§7.4, §9)
10. Provide downloadable reports for demonstrations and academic evaluation. ✔

---

## 3. Scope

**In scope — core.** Historical OHLCV retrieval, technical indicators, six comparable
forecasting algorithms, chronological and walk-forward evaluation, interactive charts,
watchlists, prediction history, CSV and PDF export.

**In scope — advanced.** Prediction intervals, SHAP explainability, strategy backtesting,
automatic model selection, model leaderboard persisted to the database, and an optional
alternative-data feature group built from NSE bulk-deal disclosures.

**Out of scope.** Live brokerage integration, order execution, guaranteed-profit strategies,
intraday or real-time tick data, and any claim of a guaranteed future price.

**Deferred (documented, not implemented).** Financial news sentiment analysis (FR-11) and
multi-stock side-by-side comparison on one screen (FR-14). See §21.

---

## 4. Functional requirements

| ID | Requirement | Priority | Status | Where |
| --- | --- | --- | --- | --- |
| FR-01 | Users can search for a stock using its ticker symbol | Must | ✔ | `GET /stocks/search`, `TickerSearch.tsx` |
| FR-02 | The system retrieves and displays historical OHLCV data | Must | ✔ | `GET /stocks/{t}/history`, `data/providers.py` |
| FR-03 | Users can select historical date ranges and forecast horizons | Must | ✔ | `workspace.tsx`, `days` / `horizon` parameters |
| FR-04 | The system cleans data and calculates technical indicators | Must | ✔ | `_tidy()`, `ml/features.py` |
| FR-05 | The system trains or loads supported ML models and generates forecasts | Must | ✔ | `POST /predictions`, `ml/pipeline.py` |
| FR-06 | Users can compare model performance using common metrics | Must | ✔ | `POST /models/compare`, `ModelComparison.tsx` |
| FR-07 | Interactive charts display actual and predicted prices | Must | ✔ | `components/charts.tsx` |
| FR-08 | Users can create a watchlist and save prediction runs | Should | ✔ | `/watchlist`, `/predictions` |
| FR-09 | Users can download prediction data and reports | Should | ✔ | `GET /reports/{id}?format=csv\|pdf` |
| FR-10 | The system displays uncertainty estimates and model limitations | Should | ✔ | `_build_forecast()`, `LIMITATIONS` |
| FR-11 | The system analyses financial news sentiment | Could | ✘ not implemented | §21 |
| FR-12 | The system provides explainability for selected model predictions | Should | ✔ | SHAP in `models/tabular.py` |
| FR-13 | The system provides historical strategy backtesting | Could | ✔ | `POST /backtest`, `ml/backtest.py` |
| FR-14 | Users can compare two or more stocks | Should | ✘ partial — watchlist shows several stocks, but there is no side-by-side analysis view | §21 |
| FR-15 | The system ingests NSE bulk-deal disclosures, displays them, and can use them as model features | Should | ✔ | `data/bulk_deals.py`, `ml/bulk_features.py`, `GET /bulk-deals/{ticker}` |

Priority definitions: **Must** = required for the first working version; **Should** =
important enhancement; **Could** = optional advanced feature.

---

## 5. Technology stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Frontend | React 19, TypeScript 5.9, Vite 8, Tailwind CSS 4, Recharts 3, Axios | Dashboard, search, interactive charts, model comparison |
| Backend | FastAPI, Pydantic v2, Uvicorn, SQLAlchemy 2, Alembic | REST API, validation, orchestration, persistence |
| AI/ML | pandas, NumPy, scikit-learn, statsmodels, XGBoost, PyTorch, SHAP | Preprocessing, feature engineering, statistical and deep models, explainability |
| Database | SQLite (default) / PostgreSQL 16 | Users, watchlists, runs, model versions, evaluations |
| Reporting | Matplotlib | PDF report generation |
| Tooling | pytest, ruff, GitHub Actions, Docker Compose | Tests, linting, CI, PostgreSQL for deployment |

Exact pinned versions are in `backend/requirements.txt` and `frontend/package.json`.

### 5.1 External data sources

- **Primary:** Yahoo Finance historical prices through the `yfinance` library, subject to its
  availability, rate limits and terms of use.
- **Offline/alternative:** a CSV provider (`CsvProvider`) reading `data/offline/<TICKER>.csv`,
  used for demonstrations without internet access and by the entire test suite.
- **Alternative data:** NSE bulk-deal disclosures, loaded from CSV exports in
  `data/bulk_deals/`. Every `*.csv` in that directory is parsed and concatenated, so further
  periods can be added without code changes. See §7.6.
- Adding a provider such as Alpha Vantage means implementing one class with a `fetch()` method
  and registering it in `get_provider()`; nothing else in the system changes.

Every response that carries prices also carries its **source and retrieval timestamp**, and
the UI displays both. Historical prices may be delayed, incomplete, adjusted for corporate
actions, or unavailable for some symbols; when a provider fails, the API returns an error or
a clearly-labelled cached download, and never fabricates prices.

---

## 6. Artificial intelligence and algorithm selection

### 6.1 Implemented algorithms

| # | Model | Family | Role |
| --- | --- | --- | --- |
| 1 | **Naive (last value)** | baseline | Carries the last adjusted close forward. The reference every other model must beat. |
| 2 | **Linear Regression** (ridge-regularised) | statistical | Interpretable baseline on lagged returns and indicators, not on the calendar date. |
| 3 | **ARIMA** | statistical | Order selected by AIC over a fixed grid, fitted on log prices with a drift term. |
| 4 | **Random Forest** | machine learning | Nonlinear interactions across the engineered features; native multi-output. |
| 5 | **XGBoost** | machine learning | Gradient boosting, one booster per horizon step; shallow trees and a low learning rate. |
| 6 | **LSTM** | deep learning | Recurrent network over a 30-session window of features, multi-step output head. |
| 7 | GRU | deep learning | Optional; same training loop with simpler gating. |
| 8 | SVR | machine learning | Optional; RBF-kernel regression on standardised features. |

Models 1–6 form the default evaluated set. All eight are resolved through a **registry
whitelist**, so an API request can never load an arbitrary class or artifact path.

### 6.2 Methods

| Method | Status |
| --- | --- |
| Naive / last-value forecast | ✔ implemented, always evaluated |
| Walk-forward validation | ✔ implemented (`walk_forward_evaluate`) |
| Empirical prediction intervals | ✔ implemented (quantiles of validation residuals) |
| SHAP explanations | ✔ implemented for tree models |
| Automatic model selection | ✔ implemented (lowest validation MAE) |
| Prophet | ✘ not implemented — ARIMA already covers the statistical benchmark |
| FinBERT news sentiment | ✘ not implemented (§21) |

> **A model is not called "most accurate" until it has been measured on unseen, chronologically
> later data and compared with the naive baseline.** The dashboard presents the skill ratio
> (`model MAE ÷ naive MAE`) next to every result for exactly this reason.

---

## 7. Data preprocessing and feature engineering

### 7.1 Input data

`Date, Open, High, Low, Close, Adjusted Close, Volume` — the adjusted close is the modelling
target because it accounts for splits and dividends.

### 7.2 Pipeline

1. **Fetch** OHLCV from the configured provider (cached with a one-hour TTL).
2. **Clean and validate** — normalise column names and timezone, drop duplicate sessions,
   drop rows with missing or non-positive prices (an in-progress session arrives as NaN),
   sort chronologically, count calendar gaps, and report every correction as a warning
   attached to the response.
3. **Engineer features** — see §7.3.
4. **Split chronologically** — 70% train, 15% validation, 15% test, never shuffled.
5. **Fit and tune** — models are fitted on training rows; scalers are fitted on training rows
   only; ARIMA orders are chosen by AIC on training rows only.
6. **Evaluate and forecast** — walk-forward through validation and test, then refit on all
   available history to produce the live forecast.

### 7.3 Features and indicators

SMA(5/10/20/50) and EMA(12/26) as ratios to the current close, RSI(14) (Wilder), MACD with
signal and histogram (normalised by price), lagged log returns (1, 2, 3, 5, 10 sessions),
10- and 20-session momentum, 10- and 20-session realised volatility, intraday range, close
position within the day's range, overnight gap, volume change and volume relative to its
20-session average — 24 features in total, plus the optional ten of §7.6.

Features are deliberately **scale-free** (ratios and returns rather than price levels), so a
model cannot memorise a price range and so the same feature definition works across stocks
and currencies.

### 7.4 The leakage rule

> Every feature on row *t* is computed only from sessions up to and including *t*, and every
> target describes sessions strictly after *t*.

This is enforced in `ml/features.py` and verified by two tests:

- `test_features_do_not_change_when_future_rows_are_removed` — recomputing the features from a
  truncated history produces byte-identical values for the shared rows.
- `test_walk_forward_never_shows_a_model_a_future_session` — a spy model records the last index
  of every frame it is handed during fitting and prediction; none ever exceeds the forecast origin.

### 7.5 Prediction target

The learning target is the **cumulative log return over the next *h* sessions**, converted
back to a price for display:

```
y_h(t) = log( AdjClose(t+h) / AdjClose(t) )      price = AdjClose(t) · exp(ŷ_h)
```

Predicting the price level directly lets any model score well by echoing the last close,
which hides whether it learned anything. Returns make the comparison with the naive baseline
meaningful, keep forecasts positive, and let different model families be scored on identical
quantities.

### 7.6 Optional feature group: NSE bulk deals

A bulk deal is a single client buying or selling more than 0.5% of a company's listed equity
in one session. NSE discloses the list **after the close** of that session, so a forecast made
after the close may use the deals disclosed for that same session; the lag is configurable
(`bulk_deals_publication_lag`) for a stricter assumption.

Ten features are derived per session: whether a deal occurred, how many, net and gross
disclosed quantity relative to the session's volume, disclosed turnover relative to the
session's turnover, the buy/sell imbalance, the premium of the volume-weighted deal price to
the close, 5- and 20-session rolling net pressure, and the number of sessions since the last
disclosure.

Three properties of this dataset drive the design:

1. **It is sparse.** The median symbol appears on two sessions. Coverage is measured per
   ticker and the group is only used when there are at least `bulk_deals_min_sessions`
   disclosed sessions.
2. **Large caps never appear.** Moving 0.5% of Reliance or TCS in one deal is impractical, so
   the symbols this module can model are small and mid caps.
3. **It covers a fixed window.** Outside that window "no deal" means "not recorded", so the
   dataset is **clipped** to the covered period rather than padded with zeros. Padding would
   teach a model that bulk deals began the day the export did. The clipped window is shorter
   than the price-only window, which the API, the UI and the report all state.

The group is tri-state per request: required, never, or — the default — used when coverage is
sufficient.

---

## 8. Prediction methodology

### 8.1 Horizons

1, 5, 10 or 30 trading sessions, chosen by the user. Forecasts are **direct** — one model per
horizon step rather than a recursive chain — so a 30-session forecast does not compound its
own error. The interface labels trading sessions, not calendar days; target dates skip
weekends but not exchange holidays, and are documented as estimates of the *n*-th future session.

### 8.2 Model comparison

All models are evaluated on the same splits, the same forecast origins, the same horizon and
the same target. Cheap models are refitted on the expanding window at every origin; ARIMA and
the recurrent models are fitted once per segment and rolled forward with newly observed
sessions, and the number of fits performed is reported with the result.

**Auto-select** chooses the model with the lowest one-session-ahead MAE **on the validation
window**. The test window is never used for selection, which is what keeps the reported test
metrics an honest estimate of unseen performance.

### 8.3 Forecast output

Every forecast reports: ticker and company name, latest available price with its timestamp,
target date, predicted price, absolute and percentage change, selected model and horizon,
the measured evaluation metrics, a 95% prediction interval where one can be estimated, the
data source and retrieval time, the model's limitations, and a disclaimer.

---

## 9. Model evaluation and performance metrics

| Metric | Meaning | Direction |
| --- | --- | --- |
| MAE | Mean absolute error in price units | lower |
| RMSE | Root mean squared error; penalises large misses | lower |
| MAPE | Mean absolute percentage error (skips near-zero actuals) | lower, with caveats |
| Directional accuracy | Share of forecasts that predicted the right direction | higher; ~0.5 is uninformative |
| R² | Variance explained relative to the test window's mean | higher; can be negative |
| **MAE / naive** | Skill ratio against the baseline | **below 1.0 to add value** |

Directional accuracy excludes pairs where either side is exactly flat: a session that closed
unchanged has no direction to predict, and a flat forecast makes no directional claim. The
naive baseline is flat by construction, so its directional accuracy is reported as undefined
rather than as 0%.

### 9.1 Validation strategy

Chronological 70/15/15 split with no shuffling, then walk-forward origins through the
validation and test segments. Origins are thinned by a configurable step so that a long
history stays affordable to evaluate, and the most recent origins are always retained.
Scalers, feature statistics and model-order selection are fitted only on data preceding each
origin.

### 9.2 Measured results

Real results from `scripts/run_experiments.py` are in
[`results/RESULTS.md`](results/RESULTS.md). Summarised for AAPL, MSFT and RELIANCE.NS over a
three-year window with a five-session horizon: **most models did not beat the naive baseline
on the test window**, directional accuracy clustered near 0.5, and the LSTM was the worst
model on one of the three stocks. This is the expected outcome for daily equity prices, and
reporting it is the purpose of the harness.

Whether any of those differences exceed chance is tested in
[`results/SIGNIFICANCE.md`](results/SIGNIFICANCE.md). Scored at every test session, no model's
error was reliably lower than the naive forecast's. One exploratory lead — the LSTM's
directional accuracy of 0.565 on three stocks — was then tested as a single pre-registered
hypothesis on 13 stocks never used elsewhere in the project, and was **not confirmed**
(0.503 against 0.487 for always predicting "up", McNemar p = 0.106).

The bulk-deal feature group was evaluated separately as a controlled ablation (same rows,
same splits, same origins, with and without the ten columns); see
[`results/BULK_DEALS.md`](results/BULK_DEALS.md).

---

## 10. Advanced features

- **10.1 Interactive dashboard** — price, volume, moving averages, RSI and MACD on separate
  panels (one measure per axis), with crosshair tooltips and a data-table view of every chart.
- **10.2 Model leaderboard** — ranked by validation MAE, showing test metrics, skill against
  the baseline, fit time and the evaluation period; persisted to the database so rankings can
  be shown without retraining.
- **10.3 Explainable AI** — SHAP `TreeExplainer` contributions for the next-session prediction
  of the tree models, presented as contributions of the model and explicitly *not* as evidence
  of a causal effect on the market.
- **10.4 News sentiment** — not implemented (§21).
- **10.5 Backtesting** — a long/flat rule driven by the one-session-ahead forecast, with a
  configurable entry threshold and one-way transaction cost, compared against buy-and-hold,
  reporting total and annualised return, volatility, Sharpe ratio, maximum drawdown, number of
  trades and time in market.
- **10.6 Report generation** — CSV (forecasts plus every evaluation row) and a one-page PDF
  (chart, forecast table, metrics, methodology, disclaimer).
- **10.7 Alternative data: NSE bulk deals** — disclosed large trades are shown per stock (who
  traded, which side, what size, at what price) and can be fed to the models as a feature
  group (§7.6). Its contribution is measured by ablation rather than assumed, and the
  dashboard labels which feature set produced every forecast.

---

## 11. System architecture

Three-tier, with the ML engine inside the backend service. See
[architecture.md](architecture.md) for diagrams.

```
Presentation   React + TypeScript     dashboard · search · charts · comparison · watchlists
      │ HTTPS/JSON
Application    FastAPI                validation · orchestration · auth · evaluation · reports
      │                                        │
      │                                   ML engine (pandas, scikit-learn, statsmodels,
      │                                   XGBoost, PyTorch, SHAP)
Data           SQLite / PostgreSQL     users · watchlists · runs · model versions · evaluations
               File cache + artifacts  cached OHLCV · fitted model artifacts
               External providers      yfinance (primary) · CSV (offline)
```

**Component responsibilities.** The frontend collects input, renders results and validates
simple fields. The API validates every request, coordinates retrieval and model execution and
returns structured responses. The ML module owns preprocessing, feature engineering, training,
forecasting and evaluation. The provider module isolates external APIs behind one interface.
The database stores persistent records. Model artifacts are versioned files on disk.

The frontend never holds a provider API key and never performs training directly.

---

## 12. Database design

### 12.1 Tables

| Table | Columns |
| --- | --- |
| `users` | id, email, password_hash, is_active, created_at |
| `watchlists` | id, user_id → users, ticker, note, created_at · unique(user_id, ticker) |
| `model_versions` | id, model_name, version, ticker, target, horizon, features, hyperparameters, train_start, train_end, artifact_path, training_seconds, created_at |
| `prediction_runs` | id, user_id → users, ticker, model_name, model_version_id → model_versions, forecast_horizon, history_days, reference_date, reference_price, data_source, data_retrieved_at, auto_selected, status, error_message, runtime_seconds, created_at |
| `prediction_results` | id, run_id → prediction_runs, step, target_date, predicted_price, lower_bound, upper_bound |
| `model_evaluations` | id, model_version_id, run_id, model_name, ticker, segment, horizon, period_start, period_end, n_observations, mae, rmse, mape, directional_accuracy, r2, mae_vs_naive, evaluated_at |
| `price_bars` | id, ticker, bar_date, open, high, low, close, adj_close, volume, source, retrieved_at · unique(ticker, bar_date) |

### 12.2 Relationships

One user has many watchlist entries and many prediction runs. One prediction run produces
many forecast rows and many evaluation rows. One model version has many evaluations and may
back many runs. Deleting a user cascades to their watchlist and runs.

Retrieved price history is cached as CSV files with a TTL; `price_bars` is available for
database-backed caching when that is preferred.

---

## 13. REST API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Service and configuration status |
| POST | `/api/v1/auth/register` | Create an account, returns a JWT |
| POST | `/api/v1/auth/login` | Sign in, returns a JWT |
| GET | `/api/v1/auth/me` | Current user |
| GET | `/api/v1/stocks/search?q=` | Search available stocks |
| GET | `/api/v1/stocks/{ticker}/history` | Historical OHLCV with provenance |
| GET | `/api/v1/stocks/{ticker}/indicators` | Technical indicators with definitions |
| GET | `/api/v1/bulk-deals/symbols` | Symbols ranked by disclosed sessions |
| GET | `/api/v1/bulk-deals/{ticker}` | Disclosed bulk deals and daily aggregates |
| POST | `/api/v1/predictions` | Generate a forecast (`model: "auto"` selects) |
| GET | `/api/v1/predictions` | Prediction history (authenticated) |
| GET | `/api/v1/predictions/{id}` | Retrieve a saved prediction |
| GET | `/api/v1/models` | Algorithm catalogue with availability |
| POST | `/api/v1/models/compare` | Evaluate several models on identical windows |
| GET | `/api/v1/models/leaderboard` | Stored rankings |
| GET | `/api/v1/watchlist` | Watchlist with latest prices |
| POST | `/api/v1/watchlist` | Add a stock |
| DELETE | `/api/v1/watchlist/{id}` | Remove an entry |
| POST | `/api/v1/backtest` | Simulate a model-driven rule |
| GET | `/api/v1/reports/{id}?format=csv\|pdf` | Download a forecast report |
| POST | `/api/v1/reports/comparison` | Download a leaderboard as CSV |

Interactive documentation is generated by FastAPI at `/docs`.

**Example request**

```json
POST /api/v1/predictions
{ "ticker": "AAPL", "model": "xgboost", "horizon": 5, "history_days": 1095 }
```

**Example response** (abbreviated; values come from the live pipeline, not from this document)

```json
{
  "ticker": "AAPL", "model": "xgboost", "forecast_horizon": 5,
  "reference_date": "2026-10-20", "reference_price": 339.50,
  "forecasts": [{ "step": 1, "target_date": "2026-10-21", "predicted_price": 338.89,
                  "lower_bound": 329.94, "upper_bound": 342.00, "change_percent": -0.18 }],
  "metrics": { "mae": 4.32, "rmse": 5.36, "mape": 1.36, "directional_accuracy": 0.381 },
  "mae_vs_naive_test": 1.017, "beats_naive_on_test": false,
  "metrics_source": "Walk-forward on the held-out test window",
  "disclaimer": "Forecasts are statistical estimates ... not investment advice."
}
```

---

## 14. User interface requirements

| Page | Contents |
| --- | --- |
| 1. Dashboard | Selected stock overview, 12-month chart, watchlist with latest prices, recent forecasts |
| 2. Stock analysis | Price with SMA/EMA overlays, volume, RSI, MACD, indicator definitions, history-window selector |
| 3. Prediction lab | Model selector, horizon, history window, forecast chart with prediction interval, metrics against the baseline, SHAP explanation, limitations |
| 4. Model comparison | Leaderboard, skill-against-baseline chart, actual-versus-predicted over the test window, CSV export |
| 5. Reports & history | Saved runs with CSV/PDF export, stored evaluations, strategy backtest with equity curve |

**UI/UX.** Responsive layout; explicit loading states that say what is being trained;
actionable error messages for invalid tickers, missing data and provider rate limits;
consistent currency and date formatting; tooltips explaining every metric and indicator; a
visible separation between observed history and forecast (a labelled origin line); the data
source and retrieval timestamp on every data panel; a standing disclaimer in the footer.

**Accessibility.** Keyboard navigation with a skip link and visible focus rings; combobox
semantics with arrow-key navigation in search; charts that never rely on colour alone — a
legend is always present for two or more series and every analytical chart offers a data-table
view; a colour palette checked for colour-vision separation and surface contrast in both light
and dark themes; `aria` labels on icon-only controls.

---

## 15. Non-functional requirements

| ID | Requirement | Target | Status |
| --- | --- | --- | --- |
| NFR-01 | Performance | Cached historical-data requests complete within 2 s | ✔ cache hit served from local CSV |
| NFR-02 | Forecast latency | A single lightweight model responds in under 5 s excluding slow retrieval | ✔ ~6 s measured end-to-end including a live fetch; a full six-model comparison takes minutes and is documented as the heavy path |
| NFR-03 | Reliability | Provider errors handled without crashing | ✔ `test_a_failing_model_does_not_abort_the_experiment`, stale-cache fallback |
| NFR-04 | Security | Bcrypt password hashing; no secrets in frontend code | ✔ `core/security.py` |
| NFR-05 | Maintainability | Frontend, API, data, ML separated into modules | ✔ see §11 |
| NFR-06 | Scalability | Training separable from prediction requests | ◐ experiment results are memoised and heavy endpoints rate-limited; a separate worker service is future work |
| NFR-07 | Usability | A new user can search a stock and see a forecast without specialist knowledge | ✔ guest session created automatically |
| NFR-08 | Reproducibility | Model versions, features, training and evaluation periods recorded | ✔ `model_versions`, `model_evaluations`, fixed seeds, `test_models_are_reproducible_with_the_same_seed` |
| NFR-09 | Compatibility | Modern Chrome, Edge, Firefox | ◐ standard ES2022 build; not yet verified on each browser |
| NFR-10 | Data integrity | Timestamps, missing observations and adjustments handled consistently | ✔ `_tidy()` and its tests |

These are acceptance targets measured on the development machine (Windows 11, Python 3.12,
Node 24), not vendor guarantees.

---

## 16. Security and privacy

1. JWT bearer authentication with a configurable expiry.
2. Passwords hashed with bcrypt; plaintext is never stored and the hash is never returned.
3. All request parameters validated by Pydantic, with ticker symbols checked against a
   restrictive pattern before any provider call.
4. All database access through SQLAlchemy's parameterised queries.
5. Secrets (JWT key, database URL, provider keys) read from environment variables; `.env` is
   git-ignored and `.env.example` contains no real values.
6. Watchlists and prediction runs are scoped to their owner; another user's row returns 404
   rather than 403, so the API does not confirm that it exists.
7. Rate limiting on the endpoints that train models.
8. Models resolved only through a registry whitelist — no user-supplied class names, file
   paths or serialised artifacts are ever loaded.
9. CORS restricted to the configured frontend origins.
10. Errors logged without recording credentials.

Verified by the security tests in `tests/test_api.py` (unauthorised access, cross-user access,
invalid tokens, injection-shaped tickers, identical messages for unknown email and wrong
password).

---

## 17. Assumptions and constraints

- Internet access is required to retrieve live market data; the CSV provider covers offline demonstrations.
- Providers may impose quotas, latency and licensing restrictions.
- Historical data is not live market data, and some symbols have incomplete records.
- Deep-learning models need more compute than the others; they are fitted once per evaluation
  segment to keep a full comparison affordable.
- Forecasts degrade during unusual market events and regime changes.
- The application is for academic analysis, not a trading system.

---

## 18. Testing strategy

| Test type | Coverage | File |
| --- | --- | --- |
| Unit — indicators | SMA, EMA, RSI bounds and extremes, MACD identity, return conversion, trading-day arithmetic | `test_features.py` |
| Unit — metrics | MAE, RMSE, MAPE edge cases, directional accuracy including the flat-forecast case, R², skill ratio | `test_metrics_and_splits.py` |
| Unit — splits | Ordering, disjointness, minimum sizes, origin thinning | `test_metrics_and_splits.py` |
| Data validation | Duplicate dates, missing and non-positive prices, unsorted input, missing columns, missing adjusted close, invalid and unknown tickers | `test_data_validation.py` |
| ML | Every model fits and predicts the requested horizon; reproducibility with a fixed seed; SHAP output; ARIMA order selection; save/load round-trip | `test_models.py` |
| Leakage | Features invariant to future rows; a spy model proves no future session reaches fitting or prediction | `test_features.py`, `test_pipeline.py` |
| Pipeline | Chronological splits, per-horizon metrics, interval bracketing, baseline comparison, selection ignoring the test window, failure isolation, refit policy | `test_pipeline.py` |
| API | Every endpoint, validation errors, provider failures, 404s, report downloads, rate limiting | `test_api.py` |
| Security | Unauthenticated access, cross-user isolation, invalid tokens, password handling | `test_api.py` |
| Backtest | Session-by-session coverage, benchmark reporting, cost effect, cash-only naive rule, drawdown sign | `test_backtest.py` |
| Bulk deals | Indian digit grouping, BOM and padded headers, unusable rows, aggregation arithmetic, coverage, window clipping, zero-versus-missing, publication lag, tri-state selection, and the ablation invariant that both arms share one set of rows | `test_bulk_deals.py` |

**176 tests, all passing, with no network access required.** The bulk-deal tests use a
synthetic NSE-shaped export, so they do not depend on the bundled file either.

Sample acceptance tests, all covered: a valid symbol displays history; an invalid symbol gives
an understandable error; sufficient history produces a forecast; the comparison page shows
metrics on comparable test periods; a failed provider never produces fabricated prices; a
saved prediction can be retrieved with its metadata; an evaluation run uses only data that
would have been available at the prediction time.

---

## 19. Development plan

| Week | Milestone | Status |
| --- | --- | --- |
| 1 | Requirements, repository, project skeleton | done |
| 2 | Data pipeline, cleaning, indicators, visualisation | done |
| 3 | Baseline, Linear Regression, ARIMA, Random Forest, XGBoost | done |
| 4 | LSTM, chronological validation, metrics, leaderboard | done |
| 5 | Frontend integration, charts, watchlists, history | done |
| 6 | Explainability, backtesting, reports, testing, documentation | done |

---

## 20. Deliverables

1. React frontend (five pages) ✔
2. FastAPI backend with documented endpoints ✔
3. Database schema with an Alembic migration ✔
4. Data ingestion and preprocessing module ✔
5. Eight forecasting algorithms ✔
6. Model evaluation and comparison dashboard ✔
7. Prediction history and watchlist ✔
8. Test suite and measured evaluation results ✔
9. README with setup and execution instructions ✔
10. SRS, architecture diagram, ER diagram and UML diagrams ✔
11. Experiment figures and model-performance analysis ✔ (`docs/results/`), including the
    bulk-deal feature ablation ✔ and a significance analysis with a pre-registered
    confirmation test ✔
12. Final report on algorithms, setup, results, limitations and future work — **to be written
    by the team**, using `docs/results/RESULTS.md` as its evidence base

---

## 21. Future work and known gaps

- **FR-11, news sentiment.** Not implemented. Doing it properly requires a licensed news
  source, careful publication-timestamp handling so that no headline published after the
  forecast origin can influence a forecast, and an independent evaluation of whether the
  sentiment feature improves accuracy at all. A stub would have added a feature to the
  screenshot list without adding evidence, so it was deliberately left out.
- **FR-14, multi-stock comparison.** The watchlist shows several stocks with their latest
  prices, but there is no side-by-side analysis or correlation view.
- **NFR-06.** Training runs inside the API process. Moving it to a worker queue would let the
  API stay responsive during a long comparison.
- **Bulk deals beyond one year.** The bundled export covers a single year, which caps the
  evaluation window for that feature group at roughly 245 sessions. Adding further exports to
  `data/bulk_deals/` widens it with no code change.
- **Hyperparameter tuning.** Models use sensible fixed settings. A proper time-series
  cross-validated search on the training window would be the next methodological step.
- **Browser verification (NFR-09).** The UI is type-checked, built and render-tested, but has
  not yet been clicked through in each supported browser.

---

## 22. Reading the results honestly

Daily equity returns are close to unpredictable. A system like this one is academically
valuable because of its **method**, not because of a high accuracy number, and the following
claims should never be made about it:

- *"The LSTM is the most accurate model."* — Only true if measured, on the test window, against
  the baseline. In the recorded runs it was the worst model on one of three stocks.
- *"MAPE is 1.3%, so the model is 98.7% accurate."* — A 1.3% average error on a price that
  moves ~1% per session is roughly what copying yesterday's close achieves.
- *"R² is 0.89, so the model explains 89% of the variance."* — On a price level, a high R² is
  produced by the price trend itself, not by predictive skill.

The defensible claim is this: **on the evaluated windows, these models were measured under a
leak-free protocol, and a small number of them beat the naive baseline by a small margin.**
