"""CSV and PDF report generation (FR-09, SRS 10.6).

Reports are assembled from stored run records and measured evaluations, never
from placeholder numbers, and every page repeats the forecast disclaimer.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: the API has no display

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from app.db.models import ModelEvaluation, PredictionRun  # noqa: E402
from app.ml.pipeline import DISCLAIMER  # noqa: E402

METRIC_FIELDS = ("mae", "rmse", "mape", "directional_accuracy", "r2", "mae_vs_naive")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def forecast_csv(run: PredictionRun, evaluations: Sequence[ModelEvaluation]) -> str:
    """Forecast rows plus the evaluation metrics behind them."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")

    writer.writerow(["Stock Market Price Prediction System - forecast report"])
    writer.writerow(["generated_at", datetime.now(timezone.utc).isoformat()])
    writer.writerow([])
    writer.writerow(["run_id", run.id])
    writer.writerow(["ticker", run.ticker])
    writer.writerow(["model", run.model_name])
    writer.writerow(["auto_selected", run.auto_selected])
    writer.writerow(["forecast_horizon_sessions", run.forecast_horizon])
    writer.writerow(["reference_date", run.reference_date])
    writer.writerow(["reference_price", _fmt(run.reference_price)])
    writer.writerow(["data_source", run.data_source])
    writer.writerow(["data_retrieved_at", run.data_retrieved_at])
    writer.writerow(["prediction_target", "cumulative log return, converted to price"])
    writer.writerow([])

    writer.writerow(["step", "target_date", "predicted_price", "lower_bound", "upper_bound", "change_percent"])
    for item in run.results:
        change = (
            (item.predicted_price / run.reference_price - 1.0) * 100.0 if run.reference_price else None
        )
        writer.writerow(
            [
                item.step,
                item.target_date,
                _fmt(item.predicted_price),
                _fmt(item.lower_bound),
                _fmt(item.upper_bound),
                _fmt(change, 3),
            ]
        )

    writer.writerow([])
    writer.writerow(["segment", "horizon", "period_start", "period_end", "n_observations", *METRIC_FIELDS])
    for evaluation in evaluations:
        writer.writerow(
            [
                evaluation.segment,
                evaluation.horizon,
                evaluation.period_start,
                evaluation.period_end,
                evaluation.n_observations,
                *[_fmt(getattr(evaluation, field)) for field in METRIC_FIELDS],
            ]
        )

    writer.writerow([])
    writer.writerow(["disclaimer", DISCLAIMER])
    return buffer.getvalue()


def comparison_csv(payload: dict[str, Any]) -> str:
    """Leaderboard export for the model comparison page."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Stock Market Price Prediction System - model comparison"])
    writer.writerow(["generated_at", datetime.now(timezone.utc).isoformat()])
    writer.writerow(["ticker", payload.get("ticker")])
    writer.writerow(["forecast_horizon_sessions", payload.get("forecast_horizon")])
    writer.writerow(["reference_date", payload.get("reference_date")])
    writer.writerow(["selection_rule", payload.get("selection_rule")])
    dataset = payload.get("dataset", {})
    for key in ("price_rows", "feature_rows", "n_train", "n_validation", "n_test",
                "train_start", "train_end", "validation_start", "validation_end",
                "test_start", "test_end"):
        if key in dataset:
            writer.writerow([key, dataset[key]])
    writer.writerow([])

    columns = [
        "model",
        "display_name",
        "family",
        "validation_mae",
        "validation_rmse",
        "test_mae",
        "test_rmse",
        "test_mape",
        "test_directional_accuracy",
        "test_r2",
        "mae_vs_naive_test",
        "beats_naive_on_test",
        "training_seconds",
        "is_best",
        "error",
    ]
    writer.writerow(columns)
    for row in payload.get("leaderboard", []):
        writer.writerow([_fmt(row.get(column)) for column in columns])

    writer.writerow([])
    writer.writerow(["disclaimer", payload.get("disclaimer", DISCLAIMER)])
    return buffer.getvalue()


def build_report_figure(
    run: PredictionRun,
    evaluations: Sequence[ModelEvaluation],
    history: pd.DataFrame | None = None,
) -> plt.Figure:
    """Lay out the one-page report: chart, forecast table, metrics, methodology.

    Kept separate from the PDF writer so the same page can be rendered to an
    image when checking the layout.
    """
    figure = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    figure.suptitle(
        f"{run.ticker} forecast report - {run.model_name}",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )

    axes = figure.add_axes([0.10, 0.63, 0.84, 0.26])
    if history is not None and not history.empty:
        recent = history.tail(120)
        axes.plot(recent.index, recent["adj_close"], label="Actual (adjusted close)", linewidth=1.4)
    if run.results:
        dates = [pd.Timestamp(item.target_date) for item in run.results]
        values = [item.predicted_price for item in run.results]
        if run.reference_date and run.reference_price:
            dates.insert(0, pd.Timestamp(run.reference_date))
            values.insert(0, run.reference_price)
        axes.plot(dates, values, "o--", color="#d9480f", label="Forecast", linewidth=1.4, markersize=3)
        lower = [item.lower_bound for item in run.results]
        upper = [item.upper_bound for item in run.results]
        if all(bound is not None for bound in lower + upper):
            axes.fill_between(
                dates[1:], lower, upper, color="#d9480f", alpha=0.15, label="95% interval"
            )
    axes.set_title("Recent history and forecast", fontsize=10)
    axes.set_ylabel("Price")
    axes.legend(fontsize=7, loc="upper left")
    axes.grid(alpha=0.25)
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for label in axes.get_xticklabels():
        label.set_fontsize(7)

    lines: list[str] = []
    lines.append("Run details")
    lines.append(f"  Run id: {run.id}    Ticker: {run.ticker}    Model: {run.model_name}")
    lines.append(
        f"  Horizon: {run.forecast_horizon} trading session(s)    "
        f"Auto-selected: {'yes' if run.auto_selected else 'no'}"
    )
    lines.append(f"  Reference session: {run.reference_date}  at  {_fmt(run.reference_price, 2)}")
    lines.append(f"  Data source: {run.data_source}    Retrieved: {run.data_retrieved_at}")
    lines.append("")
    lines.append("Forecast")
    lines.append("  step  target date   predicted      interval")
    for item in run.results:
        interval = (
            f"[{_fmt(item.lower_bound, 2)}, {_fmt(item.upper_bound, 2)}]"
            if item.lower_bound is not None
            else "not available"
        )
        lines.append(
            f"  {item.step:>4}  {item.target_date}   {_fmt(item.predicted_price, 2):>9}   {interval}"
        )
    lines.append("")
    lines.append("Measured performance (walk-forward, chronological windows)")
    lines.append("  segment      h  n    MAE      RMSE     MAPE%   DirAcc   vs naive")
    for evaluation in evaluations:
        lines.append(
            f"  {evaluation.segment:<11}{evaluation.horizon:>2}  {evaluation.n_observations:<4} "
            f"{_fmt(evaluation.mae, 3):<8} {_fmt(evaluation.rmse, 3):<8} "
            f"{_fmt(evaluation.mape, 2):<7} {_fmt(evaluation.directional_accuracy, 3):<8} "
            f"{_fmt(evaluation.mae_vs_naive, 3)}"
        )
    lines.append("")
    lines.append("Methodology")
    lines.append("  - Target: cumulative log return over the next h sessions, converted to a price.")
    lines.append("  - Features use only sessions up to the forecast origin (no look-ahead).")
    lines.append("  - Rows are split chronologically into train / validation / test; never shuffled.")
    lines.append("  - Model selection uses validation error only; the test window is held out.")
    lines.append("  - Intervals are empirical 2.5%/97.5% quantiles of validation residuals.")
    lines.append("")
    lines.append("Disclaimer")
    for chunk in _wrap(DISCLAIMER, 92):
        lines.append(f"  {chunk}")

    figure.text(
        0.08,
        0.58,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.4,
    )
    return figure


def forecast_pdf(
    run: PredictionRun,
    evaluations: Sequence[ModelEvaluation],
    history: pd.DataFrame | None = None,
) -> bytes:
    """Render the report page as a single-page PDF document."""
    figure = build_report_figure(run, evaluations, history)
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(figure)
    plt.close(figure)
    return buffer.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
