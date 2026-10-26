# Stock Market Price Prediction System

An AI laboratory project: a web application that retrieves historical stock data, engineers
technical features, forecasts future closing prices with six comparable algorithms, and
reports how well each one actually performed on data it never saw during training.

The point of the project is not that it predicts prices. It is that **every number it shows
is measured by a reproducible pipeline and compared against a naive baseline**, so the claim
"this model is useful" can be checked rather than asserted.

```
React + TypeScript + Tailwind + Recharts   →   FastAPI + SQLAlchemy   →   SQLite / PostgreSQL
                                                        ↓
                      pandas · scikit-learn · statsmodels · XGBoost · PyTorch · SHAP
```

---

## What is implemented

| Area | Implementation |
| --- | --- |
| Data | Yahoo Finance via `yfinance` behind a provider interface, TTL file cache, offline CSV provider, validation of duplicates / gaps / non-positive prices |
| Alternative data | NSE bulk-deal disclosures: parsed, displayed per stock, and usable as an optional model feature group whose value is measured by ablation |
| Features | SMA, EMA, RSI (Wilder), MACD, lagged log returns, momentum, realised volatility, volume features, intraday range — all leak-free by construction |
| Models | Naive baseline, Linear Regression (ridge), ARIMA, Random Forest, XGBoost, LSTM, plus optional GRU and SVR |
| Evaluation | Chronological 70/15/15 split, walk-forward origins, MAE / RMSE / MAPE / directional accuracy / R², skill ratio against the naive forecast |
| Forecasting | Direct multi-step forecasts (1–30 sessions) with empirical 95% prediction intervals |
| Explainability | SHAP contributions for the tree models |
| Backtesting | Long/flat rule on walk-forward forecasts with transaction costs, against buy-and-hold |
| Persistence | Users, watchlists, prediction runs, forecasts, model versions, evaluations (SQLite by default, PostgreSQL supported, Alembic migration included) |
| API | 21 documented REST endpoints with OpenAPI docs at `/docs` |
| Frontend | Five-page dashboard: overview, stock analysis, prediction lab, model comparison, reports & history |
| Security | Bcrypt password hashing, JWT auth, per-user data isolation, input validation, model whitelist, rate limiting, restricted CORS |
| Testing | 176 backend tests covering indicators, metrics, splits, leakage, every model, bulk-deal parsing and features, the API and data-validation edge cases |

---

## Quick start

Requires Python 3.12+ and Node 20.19+ (or 22.12+).

### 1. Backend

```bash
cd backend
python -m pip install -r requirements.txt
cp .env.example .env            # optional: defaults work out of the box
python -m uvicorn app.main:app --reload
```

The API is then on <http://127.0.0.1:8000>, with interactive documentation at
<http://127.0.0.1:8000/docs>. On first start it creates `data/app.db` (SQLite).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The dev server proxies `/api` to the backend, so no CORS setup
is needed for local work.

### 3. First look

1. Search for a ticker (`AAPL`, `TCS.NS`, `RELIANCE.NS`, `^NSEI`) in the header.
2. **Stock analysis** shows price, volume, moving averages, RSI, MACD and — for NSE symbols
   that have them — disclosed bulk deals.
3. **Prediction lab** runs a forecast. Start with a single model (fast); "Auto-select" trains
   the whole default set and takes a few minutes the first time for a new stock.
4. **Model comparison** evaluates every model on identical windows and ranks them.
5. **Reports & history** exports CSV/PDF reports and runs the strategy backtest.

---

## Running the tests

```bash
cd backend
python -m pytest            # 176 tests, no network access required
python -m ruff check app tests
```

```bash
cd frontend
npm run typecheck
npm run build
```

The backend suite is deterministic and offline: it generates a synthetic price series and
serves it through the CSV provider, so results do not depend on a market data feed.

---

## Reproducing the experimental results

```bash
python scripts/run_experiments.py --tickers AAPL MSFT RELIANCE.NS --horizon 5
```

This writes measured leaderboards, charts and a Markdown summary into [docs/results](docs/results):

- [`docs/results/RESULTS.md`](docs/results/RESULTS.md) — the tables quoted in the report
- `*_leaderboard.csv`, `*_experiment.json` — raw numbers
- `*_actual_vs_predicted.png`, `*_forecast.png` — figures

Everything the final report claims should come from this script, not from a hand-typed number.

### Statistical significance

```bash
python scripts/significance_check.py exploratory   # all models, 3 stocks, every test session
python scripts/significance_check.py confirm       # pre-registered follow-up on 13 fresh stocks
```

Asks whether any model's edge over the naive forecast is larger than chance (Diebold-Mariano
for error, binomial and exact McNemar for direction, Bonferroni-corrected across models), then
tests the one lead it found on stocks never used anywhere else in the project. Results:
[`docs/results/SIGNIFICANCE.md`](docs/results/SIGNIFICANCE.md).

### The bulk-deal feature ablation

```bash
python scripts/bulk_deals_ablation.py --list-symbols   # which symbols have enough disclosures
python scripts/bulk_deals_ablation.py                  # run the controlled comparison
```

This answers "do NSE bulk deals improve the forecasts?" by running each model twice over
**identical rows, splits and forecast origins**, with and without the ten bulk-deal columns,
and writes [`docs/results/BULK_DEALS.md`](docs/results/BULK_DEALS.md). It also reports how much
of each tree model's feature importance lands on those columns, so "ignored the features" and
"used them and they did not help" can be told apart.

---

## Project layout

```
backend/
  app/
    api/v1/        REST routers (stocks, predictions, models, watchlist, backtest, reports, auth)
    core/          settings and security helpers
    data/          provider interface, yfinance and CSV providers, cache, NSE bulk deals
    db/            SQLAlchemy models and session
    ml/            features, bulk-deal features, splits, metrics, models/, pipeline, backtest, registry
    reports/       CSV and PDF report builders
    schemas/       Pydantic request/response models
    services/      orchestration, persistence, rate limiting
  alembic/         PostgreSQL migration
  tests/           176 tests
frontend/
  src/
    components/    layout, charts, shared UI
    lib/           API client, types, formatting, theme, session, workspace state
    pages/         the five dashboard pages
docs/
  SRS.md           software requirements specification
  architecture.md  architecture, ER diagram, sequence and class diagrams
  results/         measured experiment output, including BULK_DEALS.md
data/
  bulk_deals/      NSE bulk-deal CSV exports (drop further periods in here)
scripts/
  run_experiments.py
  bulk_deals_ablation.py
  significance_check.py
```

---

## Using PostgreSQL instead of SQLite

```bash
# backend/.env
DATABASE_URL=postgresql+psycopg://stockpred:stockpred@localhost:5432/stockpred
```

```bash
docker compose up -d db      # optional: starts PostgreSQL 16
cd backend && python -m alembic upgrade head
```

---

## Notes and limitations

- **Daily stock returns are close to unpredictable.** In the runs recorded in
  `docs/results/RESULTS.md`, most models do *not* beat the naive last-value forecast, and
  directional accuracy sits near 50%. That is the expected result for this problem, and
  reporting it is the point of the evaluation harness.
- Forecast target dates skip weekends but not exchange holidays, so a target date is an
  estimate of the n-th future session.
- Prediction intervals are empirical quantiles of validation residuals; they assume the error
  spread stays similar, which breaks during regime changes.
- ARIMA and the recurrent models are fitted once per evaluation segment rather than at every
  origin, for runtime reasons. The harness reports the number of fits it performed.
- News sentiment (FR-11) is **not** implemented; it is documented as future work in the SRS
  rather than stubbed out.
- The NSE bulk-deal export covers one year, and large-cap stocks essentially never appear in
  it (a bulk deal needs 0.5% of a company's equity in a single trade). When that feature group
  is enabled the dataset is clipped to the disclosure window, so its evaluation window is
  about 245 sessions instead of ~700, and the UI says so on every result.
- The data provider is subject to Yahoo Finance availability, rate limits and terms of use.

**This is an educational project. Forecasts are statistical estimates from historical data,
they can be wrong, and nothing in this repository is investment advice.**
