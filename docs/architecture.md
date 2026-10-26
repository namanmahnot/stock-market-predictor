# Architecture and design documents

Diagrams are written in Mermaid, which GitHub and most Markdown viewers render directly.
Companion documents: [SRS.md](SRS.md) · [results/RESULTS.md](results/RESULTS.md)

---

## 1. System architecture

```mermaid
flowchart TB
    subgraph presentation["Presentation tier — React 19 + TypeScript + Vite"]
        UI["Dashboard · Stock analysis · Prediction lab<br/>Model comparison · Reports &amp; history"]
        CLIENT["Typed API client (Axios)<br/>JWT attached per request"]
        UI --> CLIENT
    end

    subgraph application["Application tier — FastAPI"]
        ROUTERS["Routers<br/>auth · stocks · predictions · models<br/>watchlist · backtest · reports"]
        SCHEMAS["Pydantic validation<br/>ticker pattern · horizon bounds · model whitelist"]
        SERVICES["Services<br/>experiment cache · persistence · rate limiting"]
        ROUTERS --> SCHEMAS --> SERVICES
    end

    subgraph ml["ML engine (in-process)"]
        FEATURES["features.py<br/>indicators · supervised matrix"]
        BFEAT["bulk_features.py<br/>optional NSE disclosure group"]
        PIPELINE["pipeline.py<br/>splits · walk-forward · selection · intervals"]
        REGISTRY["registry.py<br/>8 whitelisted algorithms"]
        EXPLAIN["SHAP · backtest"]
        FEATURES --> PIPELINE --> REGISTRY
        BFEAT --> PIPELINE
        PIPELINE --> EXPLAIN
    end

    subgraph data["Data tier"]
        DB[("SQLite / PostgreSQL<br/>users · watchlists · runs<br/>model_versions · evaluations")]
        CACHE[("File cache<br/>OHLCV CSV, 1 h TTL")]
        ARTIFACTS[("Model artifacts<br/>joblib")]
    end

    subgraph external["External providers"]
        YF["Yahoo Finance (yfinance)"]
        CSVP["Offline CSV provider"]
        BULK["NSE bulk-deal CSV exports<br/>data/bulk_deals/"]
    end

    CLIENT -->|"HTTP/JSON · /api/v1"| ROUTERS
    SERVICES --> PIPELINE
    SERVICES --> DB
    SERVICES --> ARTIFACTS
    PIPELINE --> FEATURES
    SERVICES -->|"provider interface"| CACHE
    CACHE --> YF
    CACHE --> CSVP
    BFEAT --> BULK
```

The ML engine runs inside the backend process. Because it is reached only through
`services/forecast_service.py`, moving it behind a task queue later would not change the API
contract or the frontend.

---

## 2. Entity-relationship diagram

```mermaid
erDiagram
    USERS ||--o{ WATCHLISTS : "follows"
    USERS ||--o{ PREDICTION_RUNS : "requests"
    PREDICTION_RUNS ||--o{ PREDICTION_RESULTS : "produces"
    PREDICTION_RUNS ||--o{ MODEL_EVALUATIONS : "measured by"
    MODEL_VERSIONS ||--o{ MODEL_EVALUATIONS : "scored as"
    MODEL_VERSIONS ||--o{ PREDICTION_RUNS : "used by"

    USERS {
        int id PK
        string email UK
        string password_hash
        bool is_active
        datetime created_at
    }
    WATCHLISTS {
        int id PK
        int user_id FK
        string ticker
        string note
        datetime created_at
    }
    MODEL_VERSIONS {
        int id PK
        string model_name
        string version
        string ticker
        string target
        int horizon
        text features
        text hyperparameters
        date train_start
        date train_end
        string artifact_path
        float training_seconds
        datetime created_at
    }
    PREDICTION_RUNS {
        int id PK
        int user_id FK
        int model_version_id FK
        string ticker
        string model_name
        int forecast_horizon
        int history_days
        date reference_date
        float reference_price
        string data_source
        datetime data_retrieved_at
        bool auto_selected
        string status
        text error_message
        float runtime_seconds
        datetime created_at
    }
    PREDICTION_RESULTS {
        int id PK
        int run_id FK
        int step
        date target_date
        float predicted_price
        float lower_bound
        float upper_bound
    }
    MODEL_EVALUATIONS {
        int id PK
        int model_version_id FK
        int run_id FK
        string model_name
        string ticker
        string segment
        int horizon
        date period_start
        date period_end
        int n_observations
        float mae
        float rmse
        float mape
        float directional_accuracy
        float r2
        float mae_vs_naive
        datetime evaluated_at
    }
    PRICE_BARS {
        int id PK
        string ticker
        date bar_date
        float open
        float high
        float low
        float close
        float adj_close
        float volume
        string source
        datetime retrieved_at
    }
```

`PRICE_BARS` is a standalone optional cache table; the running system caches OHLCV as CSV
files instead, so it has no foreign keys.

---

## 3. Sequence: generating a forecast

```mermaid
sequenceDiagram
    actor User
    participant UI as Prediction lab (React)
    participant API as FastAPI router
    participant SVC as forecast_service
    participant PROV as Data provider
    participant ML as Pipeline + models
    participant DB as Database

    User->>UI: choose model, horizon, history
    UI->>API: POST /api/v1/predictions
    API->>API: validate ticker, horizon, model whitelist
    API->>SVC: get_experiment(ticker, horizon, models)

    alt cached for this ticker, horizon and trading session
        SVC-->>API: cached ExperimentResult
    else fresh run
        SVC->>PROV: load_history(ticker, days)
        PROV->>PROV: cache hit? else fetch, then clean and validate
        PROV-->>SVC: PriceHistory + source + timestamp + warnings
        SVC->>ML: run_experiment(...)
        ML->>ML: engineer features (no future data)
        ML->>ML: chronological split 70/15/15
        loop each model
            ML->>ML: walk-forward validation → metrics + residuals
            ML->>ML: walk-forward test → held-out metrics
            ML->>ML: refit on all history → forecast + interval
        end
        ML->>ML: compare to naive, select on validation MAE
        ML-->>SVC: ExperimentResult
    end

    SVC->>DB: persist model version, run, forecasts, evaluations
    DB-->>SVC: run_id
    SVC-->>API: prediction payload
    API-->>UI: forecast + metrics + interval + provenance + disclaimer
    UI-->>User: chart, metrics against the baseline, limitations
```

---

## 4. Class diagram: the model layer

```mermaid
classDiagram
    class BaseForecaster {
        <<abstract>>
        +str name
        +str display_name
        +str family
        +str refit_cost
        +bool supports_explanations
        +int horizon
        +is_available() tuple
        +fit(frame) BaseForecaster
        +predict(frame) ndarray
        +explain(frame) dict
        +save(path) Path
        +load(path) BaseForecaster
        #_fit(frame)*
        #_predict(frame)* ndarray
    }

    class NaiveForecaster {
        predicts zero log return
    }
    class ArimaForecaster {
        +tuple order_
        AIC grid selection
    }
    class TabularForecaster {
        <<abstract>>
        +bool scale_features
        #_build_estimator()*
        +explain(frame) dict
    }
    class LinearRegressionForecaster {
        ridge, standardised
        +coefficients()
    }
    class SVRForecaster {
        RBF kernel
    }
    class RandomForestForecaster {
        native multi-output
        +feature_importances()
    }
    class XGBoostForecaster {
        one booster per step
        +feature_importances()
    }
    class RecurrentForecaster {
        <<abstract>>
        +int window
        +str cell
        early stopping
    }
    class LSTMForecaster
    class GRUForecaster

    BaseForecaster <|-- NaiveForecaster
    BaseForecaster <|-- ArimaForecaster
    BaseForecaster <|-- TabularForecaster
    BaseForecaster <|-- RecurrentForecaster
    TabularForecaster <|-- LinearRegressionForecaster
    TabularForecaster <|-- SVRForecaster
    TabularForecaster <|-- RandomForestForecaster
    TabularForecaster <|-- XGBoostForecaster
    RecurrentForecaster <|-- LSTMForecaster
    RecurrentForecaster <|-- GRUForecaster
```

Every model receives the feature frame up to the forecast origin and returns cumulative log
returns for `h = 1..horizon`. Because the interface is identical, the evaluation harness can
score a naive rule, a statistical model and a neural network on exactly the same quantities.

---

## 5. Data and evaluation flow

```mermaid
flowchart LR
    RAW["Raw OHLCV"] --> CLEAN["Clean &amp; validate<br/>duplicates · NaN · non-positive<br/>sort · gap report"]
    CLEAN --> FEAT["24 scale-free features<br/>computed from sessions ≤ t"]
    BULKSRC["NSE bulk deals<br/>(optional)"] --> BULKF["+10 disclosure features<br/>clipped to the covered window"]
    FEAT --> BULKF
    FEAT --> SPLIT{"Chronological split"}
    BULKF --> SPLIT
    SPLIT -->|"70%"| TRAIN["Train"]
    SPLIT -->|"15%"| VAL["Validation"]
    SPLIT -->|"15%"| TEST["Test (held out)"]

    TRAIN --> FIT["Fit model"]
    FIT --> WFV["Walk-forward over validation<br/>refit on expanding window"]
    WFV --> SELECT["Select model<br/>lowest validation MAE"]
    WFV --> RESID["Validation residuals<br/>→ 95% intervals"]
    FIT --> WFT["Walk-forward over test"]
    WFT --> REPORT["Reported metrics<br/>MAE · RMSE · MAPE · DirAcc · R²<br/>÷ naive = skill"]

    TEST -.->|"never used for selection"| SELECT
    SELECT --> FINAL["Refit on all history"]
    RESID --> FINAL
    FINAL --> FORECAST["Forecast + interval"]
```

The dotted edge is the discipline the whole project rests on: the test window supplies the
numbers that are *reported*, never the numbers that *choose* a model.

---

## 6. Request validation and error paths

```mermaid
flowchart TD
    REQ["Incoming request"] --> V1{"Ticker matches<br/>the symbol pattern?"}
    V1 -->|no| E422["422 — explains the valid format"]
    V1 -->|yes| V2{"Model in the<br/>registry whitelist?"}
    V2 -->|no| E422b["422 — lists supported models"]
    V2 -->|yes| V3{"Horizon and history<br/>within bounds?"}
    V3 -->|no| E422c["422 — field-level message"]
    V3 -->|yes| V4{"Rate limit<br/>exceeded?"}
    V4 -->|yes| E429["429 + Retry-After"]
    V4 -->|no| FETCH["Retrieve history"]
    FETCH --> V5{"Provider returned<br/>usable rows?"}
    V5 -->|no, cache available| CACHED["Serve cached download<br/>labelled as cached"]
    V5 -->|no| E502["502 — never fabricates prices"]
    V5 -->|yes| V6{"Enough sessions after<br/>indicator warm-up?"}
    V6 -->|no| E422d["422 — asks for a longer window"]
    V6 -->|yes| RUN["Run the pipeline"]
    RUN --> V7{"Model failed?"}
    V7 -->|yes| ISOLATE["Record the failure,<br/>keep the other models"]
    V7 -->|no| OK["200 — forecast, metrics, provenance, disclaimer"]
```

---

## 7. Deployment view

```mermaid
flowchart LR
    subgraph dev["Development (default)"]
        VITE["Vite dev server :5173<br/>proxies /api"]
        UVI["Uvicorn :8000<br/>--reload"]
        SQLITE[("SQLite data/app.db")]
        VITE --> UVI --> SQLITE
    end

    subgraph prod["Deployment configuration"]
        STATIC["Static build (dist/)<br/>served by any web server"]
        API2["Uvicorn workers<br/>CORS restricted to the frontend origin"]
        PG[("PostgreSQL 16<br/>docker compose up -d db<br/>alembic upgrade head")]
        STATIC --> API2 --> PG
    end

    subgraph ci["CI — GitHub Actions"]
        LINT["ruff check"]
        PYTEST["pytest (offline, 176 tests)"]
        TSC["tsc --noEmit"]
        BUILD["vite build"]
    end
```

Secrets (`JWT_SECRET`, `DATABASE_URL`, any provider key) come from the environment in both
configurations and are never baked into the frontend bundle.
