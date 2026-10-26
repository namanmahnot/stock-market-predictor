"""Walk-forward evaluation and forecasting pipeline (SRS sections 8 and 9).

The pipeline is the part of the project that makes the numbers defensible:

* rows are split chronologically, never shuffled;
* a model is only ever fitted on sessions that precede the forecast origin;
* every model is scored on the same origins, the same horizon and the same
  prediction target, and is compared against the naive baseline;
* the model used for the live forecast is refitted on all available history
  after evaluation, and its reported metrics still come from the walk-forward
  windows, so the test segment is never used to pick a model.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from app.core.config import settings
from app.data.bulk_deals import BulkDealsUnavailableError
from app.data.providers import PriceHistory
from app.ml import metrics as metric_lib
from app.ml.base import BaseForecaster, NotEnoughDataError
from app.ml.bulk_features import add_bulk_deal_features
from app.ml.features import (
    build_feature_frame,
    feature_columns,
    next_trading_days,
    prices_from_log_returns,
)
from app.ml.registry import (
    BASELINE_MODEL,
    DEFAULT_MODEL_SET,
    create_model,
    get_model_class,
)
from app.ml.splits import Split, chronological_split, walk_forward_origins

logger = logging.getLogger(__name__)


#: Pass as ``max_origins`` to evaluate every origin in a segment, which the
#: backtest needs because it trades session by session.
UNLIMITED_ORIGINS = 0

#: At or below this many rows a segment is evaluated at every origin.
SHORT_SEGMENT_ROWS = 60


class InsufficientHistoryError(ValueError):
    """Raised when a ticker has too little usable history to evaluate models."""


@dataclass
class ForecastStep:
    step: int
    target_date: date
    predicted_price: float
    lower_bound: float | None
    upper_bound: float | None
    change: float
    change_percent: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "target_date": self.target_date.isoformat(),
            "predicted_price": round(self.predicted_price, 4),
            "lower_bound": None if self.lower_bound is None else round(self.lower_bound, 4),
            "upper_bound": None if self.upper_bound is None else round(self.upper_bound, 4),
            "change": round(self.change, 4),
            "change_percent": round(self.change_percent, 4),
        }


@dataclass
class SegmentEvaluation:
    """Walk-forward results for one model on one chronological segment."""

    segment: str
    horizon: int
    n_origins: int
    period_start: date | None
    period_end: date | None
    metrics_by_horizon: dict[int, metric_lib.MetricSet]
    residual_quantiles: dict[int, tuple[float, float]]
    curve: list[dict[str, Any]] = field(default_factory=list)
    refits: int = 0

    @property
    def primary(self) -> metric_lib.MetricSet | None:
        """Metrics for the one-session-ahead forecast."""
        return self.metrics_by_horizon.get(1)

    def as_dict(self, include_curve: bool = True) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "horizon": self.horizon,
            "n_origins": self.n_origins,
            "refits": self.refits,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "metrics": self.primary.as_dict() if self.primary else None,
            "metrics_by_horizon": {
                str(h): m.as_dict() for h, m in sorted(self.metrics_by_horizon.items())
            },
            "curve": self.curve if include_curve else [],
        }


@dataclass
class ModelReport:
    name: str
    display_name: str
    family: str
    hyperparameters: dict[str, Any]
    validation: SegmentEvaluation | None = None
    test: SegmentEvaluation | None = None
    forecast: list[ForecastStep] = field(default_factory=list)
    training_seconds: float | None = None
    artifact_path: str | None = None
    error: str | None = None
    mae_vs_naive_validation: float | None = None
    mae_vs_naive_test: float | None = None
    beats_naive_on_test: bool | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self, include_curve: bool = True) -> dict[str, Any]:
        return {
            "model": self.name,
            "display_name": self.display_name,
            "family": self.family,
            "hyperparameters": self.hyperparameters,
            "training_seconds": None if self.training_seconds is None else round(self.training_seconds, 3),
            "validation": self.validation.as_dict(include_curve) if self.validation else None,
            "test": self.test.as_dict(include_curve) if self.test else None,
            "forecast": [step.as_dict() for step in self.forecast],
            "mae_vs_naive_validation": self.mae_vs_naive_validation,
            "mae_vs_naive_test": self.mae_vs_naive_test,
            "beats_naive_on_test": self.beats_naive_on_test,
            "error": self.error,
        }


@dataclass
class ExperimentResult:
    ticker: str
    company_name: str | None
    currency: str | None
    data_source: str
    data_retrieved_at: datetime
    data_warnings: list[str]
    reference_date: date
    reference_price: float
    horizon: int
    history_days: int
    n_rows: int
    feature_rows: int
    split: Split
    split_dates: dict[str, str | None]
    feature_set: str
    feature_columns: list[str]
    dataset_notes: list[str]
    bulk_deals: dict[str, Any] | None
    reports: list[ModelReport]
    best_model: str | None
    selection_rule: str
    runtime_seconds: float

    def report_for(self, model_name: str) -> ModelReport | None:
        return next((r for r in self.reports if r.name == model_name), None)

    def as_dict(self, include_curve: bool = True) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "currency": self.currency,
            "data_source": self.data_source,
            "data_retrieved_at": self.data_retrieved_at.isoformat(),
            "data_warnings": self.data_warnings,
            "reference_date": self.reference_date.isoformat(),
            "reference_price": round(self.reference_price, 4),
            "forecast_horizon": self.horizon,
            "history_days": self.history_days,
            "dataset": {
                "price_rows": self.n_rows,
                "feature_rows": self.feature_rows,
                "feature_set": self.feature_set,
                "n_features": len(self.feature_columns),
                "features": self.feature_columns,
                "notes": self.dataset_notes,
                **self.split.describe(),
                **self.split_dates,
            },
            "bulk_deals": self.bulk_deals,
            "best_model": self.best_model,
            "selection_rule": self.selection_rule,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "models": [r.as_dict(include_curve) for r in self.reports],
            "disclaimer": DISCLAIMER,
        }


DISCLAIMER = (
    "Forecasts are statistical estimates produced from historical data for educational "
    "analysis. They can be wrong, they are not investment advice, and no future price is "
    "guaranteed."
)


# ---------------------------------------------------------------------------
# dataset preparation
# ---------------------------------------------------------------------------
@dataclass
class PreparedDataset:
    """The engineered frame plus a record of how it was built."""

    frame: pd.DataFrame
    columns: list[str]
    uses_bulk_deals: bool
    notes: list[str] = field(default_factory=list)
    bulk_deals: dict[str, Any] | None = None

    @property
    def feature_set(self) -> str:
        return "price+bulk_deals" if self.uses_bulk_deals else "price_only"


def prepare_dataset(history: PriceHistory, use_bulk_deals: bool | None = False) -> PreparedDataset:
    """Engineer features and verify there is enough history to evaluate on.

    ``use_bulk_deals`` is tri-state: ``True`` requires the NSE bulk-deal feature
    group and fails if it cannot be supplied, ``False`` never uses it, and
    ``None`` means "use it when this ticker has enough disclosed sessions",
    which is what the API sends by default.
    """
    frame = build_feature_frame(history.frame)
    notes: list[str] = []
    bulk_payload: dict[str, Any] | None = None
    uses_bulk = False

    if use_bulk_deals is not False:
        try:
            result = add_bulk_deal_features(frame, history.ticker)
            if use_bulk_deals is None and not result.coverage.sufficient:
                notes.append(
                    f"Bulk-deal features skipped: {history.ticker} has only "
                    f"{result.coverage.sessions_with_deals} disclosed session(s), below the "
                    f"minimum of {settings.bulk_deals_min_sessions}."
                )
            elif use_bulk_deals is None and len(result.frame) < settings.bulk_deals_min_rows:
                notes.append(
                    f"Bulk-deal features skipped: clipping to the disclosure window would leave "
                    f"{len(result.frame)} sessions, below the minimum of {settings.bulk_deals_min_rows}."
                )
            else:
                frame = result.frame
                uses_bulk = True
                notes.extend(result.notes)
                bulk_payload = result.as_dict()
        except BulkDealsUnavailableError as exc:
            if use_bulk_deals is True:
                raise InsufficientHistoryError(str(exc)) from exc
            notes.append(f"Bulk-deal features unavailable: {exc}")

    minimum = settings.bulk_deals_min_rows if uses_bulk else settings.min_observations
    if len(frame) < minimum:
        raise InsufficientHistoryError(
            f"{history.ticker} has {len(frame)} usable sessions after indicator warm-up"
            f"{' and clipping to the bulk-deal window' if uses_bulk else ''}; "
            f"at least {minimum} are required for a chronological train/validation/test "
            "evaluation. Request a longer history window."
        )

    if uses_bulk:
        notes.append(
            "The evaluation window is short because it is limited to the bulk-deal disclosure "
            "period, so these metrics are noisier than the price-only run."
        )

    return PreparedDataset(
        frame=frame,
        columns=feature_columns(frame),
        uses_bulk_deals=uses_bulk,
        notes=notes,
        bulk_deals=bulk_payload,
    )


# ---------------------------------------------------------------------------
# walk-forward evaluation
# ---------------------------------------------------------------------------
def _fit_model(model_name: str, frame: pd.DataFrame, horizon: int, **params: Any) -> BaseForecaster:
    model = create_model(model_name, horizon=horizon, **params)
    model.fit(frame)
    return model


def walk_forward_evaluate(
    model_name: str,
    frame: pd.DataFrame,
    segment: range,
    horizon: int,
    segment_name: str,
    origin_step: int | None = None,
    refit_every: int = 1,
    max_origins: int | None = None,
    model_params: dict[str, Any] | None = None,
) -> SegmentEvaluation:
    """Roll a forecast origin through ``segment`` and score every horizon step.

    At origin ``t`` the model may only see rows ``0..t``. ``origin_step`` thins
    the origins so a long history stays affordable to evaluate. Cheap models
    are refitted on the expanding window at every origin; models marked
    ``refit_cost = "high"`` (ARIMA, LSTM, GRU) are fitted once at the first
    origin of the segment and then rolled forward with newly observed sessions.
    The number of fits is reported as ``refits``.
    """
    params = dict(model_params or {})
    model_class = get_model_class(model_name)
    if origin_step is None:
        # A short segment cannot spare any origins: thinning it would leave too
        # few residuals to estimate a prediction interval from.
        origin_step = 1 if len(segment) <= SHORT_SEGMENT_ROWS else settings.walk_forward_step
    if max_origins is None:
        max_origins = settings.max_walk_forward_origins
    origin_cap = None if max_origins <= 0 else max_origins

    prices = frame["adj_close"].to_numpy(dtype="float64")
    log_prices = np.log(prices)
    n_rows = len(frame)

    usable = [t for t in segment if t + 1 < n_rows]
    origins = (
        walk_forward_origins(
            range(usable[0], usable[-1] + 1), step=origin_step, max_origins=origin_cap
        )
        if usable
        else []
    )
    if not origins:
        return SegmentEvaluation(segment_name, horizon, 0, None, None, {}, {})

    expensive = model_class.refit_cost == "high"
    model: BaseForecaster | None = None
    refits = 0

    predicted: dict[int, list[float]] = {h: [] for h in range(1, horizon + 1)}
    actual: dict[int, list[float]] = {h: [] for h in range(1, horizon + 1)}
    reference: dict[int, list[float]] = {h: [] for h in range(1, horizon + 1)}
    residuals: dict[int, list[float]] = {h: [] for h in range(1, horizon + 1)}
    curve: list[dict[str, Any]] = []

    for position, origin in enumerate(origins):
        window = frame.iloc[: origin + 1]
        if model is None or (not expensive and position % max(refit_every, 1) == 0):
            try:
                model = _fit_model(model_name, window, horizon, **params)
                refits += 1
            except NotEnoughDataError:
                if model is None:
                    continue
            except Exception as exc:  # a single bad fit must not kill the sweep
                logger.warning("%s failed to fit at origin %s: %s", model_name, origin, exc)
                if model is None:
                    continue

        try:
            cumulative = model.predict(window)
        except Exception as exc:
            logger.warning("%s failed to predict at origin %s: %s", model_name, origin, exc)
            continue

        ref_price = float(prices[origin])
        predicted_path = prices_from_log_returns(ref_price, cumulative)
        for h in range(1, horizon + 1):
            target_row = origin + h
            if target_row >= n_rows:
                break
            predicted[h].append(float(predicted_path[h - 1]))
            actual[h].append(float(prices[target_row]))
            reference[h].append(ref_price)
            realised = float(log_prices[target_row] - log_prices[origin])
            residuals[h].append(realised - float(cumulative[h - 1]))

        if origin + 1 < n_rows:
            curve.append(
                {
                    "origin_date": frame.index[origin].date().isoformat(),
                    "target_date": frame.index[origin + 1].date().isoformat(),
                    "reference_price": round(ref_price, 4),
                    "actual": round(float(prices[origin + 1]), 4),
                    "predicted": round(float(predicted_path[0]), 4),
                }
            )

    metrics_by_horizon = {
        h: metric_lib.evaluate(np.array(actual[h]), np.array(predicted[h]), np.array(reference[h]))
        for h in range(1, horizon + 1)
        if actual[h]
    }
    quantiles = {
        h: (
            float(np.quantile(residuals[h], 0.025)),
            float(np.quantile(residuals[h], 0.975)),
        )
        for h in range(1, horizon + 1)
        if len(residuals[h]) >= 20
    }

    return SegmentEvaluation(
        segment=segment_name,
        horizon=horizon,
        n_origins=len(curve) or len(origins),
        period_start=frame.index[origins[0]].date(),
        period_end=frame.index[min(origins[-1] + 1, n_rows - 1)].date(),
        metrics_by_horizon=metrics_by_horizon,
        residual_quantiles=quantiles,
        curve=curve,
        refits=refits,
    )


# ---------------------------------------------------------------------------
# forecasting
# ---------------------------------------------------------------------------
def _build_forecast(
    model: BaseForecaster,
    frame: pd.DataFrame,
    horizon: int,
    residual_quantiles: dict[int, tuple[float, float]],
) -> list[ForecastStep]:
    reference_price = float(frame["adj_close"].iloc[-1])
    cumulative = model.predict(frame)
    prices = prices_from_log_returns(reference_price, cumulative)
    dates = next_trading_days(frame.index[-1], horizon)

    steps: list[ForecastStep] = []
    for h in range(1, horizon + 1):
        predicted = float(prices[h - 1])
        bounds = residual_quantiles.get(h)
        lower = upper = None
        if bounds is not None:
            # Empirical quantiles of validation residuals in log-return space,
            # so the interval stays multiplicative and cannot go negative.
            lower = float(reference_price * np.exp(cumulative[h - 1] + bounds[0]))
            upper = float(reference_price * np.exp(cumulative[h - 1] + bounds[1]))
        steps.append(
            ForecastStep(
                step=h,
                target_date=dates[h - 1].date(),
                predicted_price=predicted,
                lower_bound=lower,
                upper_bound=upper,
                change=predicted - reference_price,
                change_percent=(predicted / reference_price - 1.0) * 100.0,
            )
        )
    return steps


def _artifact_path(ticker: str, model_name: str, horizon: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_ticker = ticker.replace("^", "idx_").replace(".", "_")
    return str(settings.artifact_dir / f"{safe_ticker}_{model_name}_h{horizon}_{stamp}.joblib")


# ---------------------------------------------------------------------------
# experiment orchestration
# ---------------------------------------------------------------------------
def run_experiment(
    history: PriceHistory,
    horizon: int = 1,
    model_names: Iterable[str] | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    history_days: int | None = None,
    use_bulk_deals: bool | None = False,
    prepared: PreparedDataset | None = None,
    evaluate_test: bool = True,
    save_artifacts: bool = False,
    include_curves: bool = True,
) -> ExperimentResult:
    """Evaluate the requested models on one ticker and forecast the horizon.

    ``use_bulk_deals`` selects the feature set (see :func:`prepare_dataset`);
    ``prepared`` overrides it with a dataset the caller has already built.
    """
    started = time.perf_counter()
    horizon = max(1, min(int(horizon), settings.max_horizon))
    names = list(model_names or DEFAULT_MODEL_SET)
    if BASELINE_MODEL not in names:
        # The baseline is always evaluated so every model can be compared to it.
        names.append(BASELINE_MODEL)
    params_by_model = dict(model_params or {})

    # A caller can supply an already-built dataset, which is how the feature
    # ablation guarantees that both arms see exactly the same rows and splits.
    prepared = prepared or prepare_dataset(history, use_bulk_deals=use_bulk_deals)
    frame = prepared.frame
    split = chronological_split(
        len(frame), settings.train_fraction, settings.validation_fraction
    )
    split_dates = {
        "train_start": frame.index[split.train[0]].date().isoformat(),
        "train_end": frame.index[split.train[-1]].date().isoformat(),
        "validation_start": frame.index[split.validation[0]].date().isoformat(),
        "validation_end": frame.index[split.validation[-1]].date().isoformat(),
        "test_start": frame.index[split.test[0]].date().isoformat(),
        "test_end": frame.index[split.test[-1]].date().isoformat(),
    }

    reports: list[ModelReport] = []
    for name in names:
        params = params_by_model.get(name, {})
        try:
            template = create_model(name, horizon=horizon, **params)
        except Exception as exc:
            reports.append(
                ModelReport(
                    name=name,
                    display_name=name,
                    family="unknown",
                    hyperparameters=params,
                    error=str(exc),
                )
            )
            continue

        report = ModelReport(
            name=name,
            display_name=template.display_name,
            family=template.family,
            hyperparameters=template.hyperparameters(),
        )
        try:
            report.validation = walk_forward_evaluate(
                name, frame, split.validation, horizon, "validation", model_params=params
            )
            if evaluate_test:
                report.test = walk_forward_evaluate(
                    name, frame, split.test, horizon, "test", model_params=params
                )

            fit_started = time.perf_counter()
            final_model = _fit_model(name, frame, horizon, **params)
            report.training_seconds = time.perf_counter() - fit_started
            report.hyperparameters = final_model.hyperparameters()
            quantiles = report.validation.residual_quantiles if report.validation else {}
            report.forecast = _build_forecast(final_model, frame, horizon, quantiles)

            if save_artifacts:
                try:
                    path = _artifact_path(history.ticker, name, horizon)
                    final_model.save(path)
                    report.artifact_path = path
                except Exception:
                    logger.warning("Could not persist artifact for %s", name, exc_info=True)
        except NotEnoughDataError as exc:
            report.error = str(exc)
        except Exception as exc:
            logger.exception("Model %s failed on %s", name, history.ticker)
            report.error = f"{type(exc).__name__}: {exc}"
        reports.append(report)

    _attach_baseline_comparison(reports)
    best_model, rule = select_best_model(reports)

    if not include_curves:
        for report in reports:
            for segment in (report.validation, report.test):
                if segment is not None:
                    segment.curve = []

    return ExperimentResult(
        ticker=history.ticker,
        company_name=history.name,
        currency=history.currency,
        data_source=history.source,
        data_retrieved_at=history.retrieved_at,
        data_warnings=history.warnings,
        reference_date=frame.index[-1].date(),
        reference_price=float(frame["adj_close"].iloc[-1]),
        horizon=horizon,
        history_days=int(history_days or len(history.frame)),
        n_rows=len(history.frame),
        feature_rows=len(frame),
        split=split,
        split_dates=split_dates,
        feature_set=prepared.feature_set,
        feature_columns=prepared.columns,
        dataset_notes=prepared.notes,
        bulk_deals=prepared.bulk_deals,
        reports=reports,
        best_model=best_model,
        selection_rule=rule,
        runtime_seconds=time.perf_counter() - started,
    )


def _attach_baseline_comparison(reports: list[ModelReport]) -> None:
    baseline = next((r for r in reports if r.name == BASELINE_MODEL and r.ok), None)
    if baseline is None:
        return
    naive_validation = baseline.validation.primary.mae if baseline.validation and baseline.validation.primary else None
    naive_test = baseline.test.primary.mae if baseline.test and baseline.test.primary else None
    for report in reports:
        if not report.ok:
            continue
        model_validation = report.validation.primary.mae if report.validation and report.validation.primary else None
        model_test = report.test.primary.mae if report.test and report.test.primary else None
        report.mae_vs_naive_validation = metric_lib.skill_vs_naive(model_validation, naive_validation)
        report.mae_vs_naive_test = metric_lib.skill_vs_naive(model_test, naive_test)
        if report.name == BASELINE_MODEL:
            report.beats_naive_on_test = None
        elif report.mae_vs_naive_test is not None:
            report.beats_naive_on_test = report.mae_vs_naive_test < 1.0


def select_best_model(reports: list[ModelReport]) -> tuple[str | None, str]:
    """Pick the model with the lowest validation MAE for the next session.

    Selection never looks at the test segment, which is what keeps the reported
    test metrics an honest estimate of unseen performance (SRS 8.2).
    """
    rule = "Lowest one-session-ahead MAE on the chronological validation window; the test window is never used for selection."
    eligible = [
        (r.validation.primary.mae, r.name)
        for r in reports
        if r.ok and r.validation and r.validation.primary and r.validation.primary.mae is not None
    ]
    if not eligible:
        return None, rule
    eligible.sort()
    return eligible[0][1], rule


def leaderboard_rows(result: ExperimentResult) -> list[dict[str, Any]]:
    """Flat rows for the model leaderboard table (SRS 10.2)."""
    rows: list[dict[str, Any]] = []
    for report in result.reports:
        if not report.ok:
            rows.append(
                {
                    "model": report.name,
                    "display_name": report.display_name,
                    "family": report.family,
                    "error": report.error,
                }
            )
            continue
        validation = report.validation.primary if report.validation else None
        test = report.test.primary if report.test else None
        rows.append(
            {
                "model": report.name,
                "display_name": report.display_name,
                "family": report.family,
                "validation_mae": validation.mae if validation else None,
                "validation_rmse": validation.rmse if validation else None,
                "test_mae": test.mae if test else None,
                "test_rmse": test.rmse if test else None,
                "test_mape": test.mape if test else None,
                "test_directional_accuracy": test.directional_accuracy if test else None,
                "test_r2": test.r2 if test else None,
                "mae_vs_naive_test": report.mae_vs_naive_test,
                "beats_naive_on_test": report.beats_naive_on_test,
                "training_seconds": report.training_seconds,
                "is_best": report.name == result.best_model,
                "error": None,
            }
        )
    rows.sort(key=lambda row: (row.get("validation_mae") is None, row.get("validation_mae") or 0.0))
    return rows
