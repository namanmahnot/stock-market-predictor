"""Reproducible experiment runner for the project report.

Evaluates every model on one or more tickers and writes the measured numbers,
charts and a Markdown summary into docs/results. Everything the final report
quotes should come from this script rather than from a manual note, so results
can be regenerated and checked.

Examples
--------
    python scripts/run_experiments.py
    python scripts/run_experiments.py --tickers AAPL RELIANCE.NS --horizon 5
    python scripts/run_experiments.py --models naive linear_regression xgboost
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# scikit-learn emits this once per parallel tree fit when a forest is trained
# inside another parallel context; it is noise, not a problem with the fit.
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

# Windows consoles default to cp1252, which cannot print the symbols used below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.data.providers import DataUnavailableError, load_history  # noqa: E402
from app.ml.pipeline import InsufficientHistoryError, leaderboard_rows, run_experiment  # noqa: E402
from app.ml.registry import DEFAULT_MODEL_SET  # noqa: E402

DEFAULT_TICKERS = ["AAPL", "MSFT", "RELIANCE.NS"]

#: Categorical palette assigned in a fixed order and never cycled; checked for
#: colour-vision separation against the chart surface.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
GRID_ALPHA = 0.25


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def plot_actual_vs_predicted(result, output: Path) -> Path | None:
    """One panel per model: realised price against the walk-forward forecast."""
    usable = [r for r in result.reports if r.ok and r.test and r.test.curve]
    if not usable:
        return None

    rows = (len(usable) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(12, 3.1 * rows), squeeze=False)
    for index, report in enumerate(usable):
        axis = axes[index // 2][index % 2]
        curve = pd.DataFrame(report.test.curve)
        dates = pd.to_datetime(curve["target_date"])
        axis.plot(dates, curve["actual"], label="Actual", linewidth=2.0, color=INK)
        axis.plot(dates, curve["predicted"], label="Predicted", linewidth=2.0, color=PALETTE[1], linestyle="--")
        metrics = report.test.primary
        subtitle = f"MAE {_fmt(metrics.mae, 3)}  |  vs naive {_fmt(report.mae_vs_naive_test, 3)}"
        axis.set_title(f"{report.display_name}\n{subtitle}", fontsize=9)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=GRID_ALPHA)
        axis.legend(fontsize=7)
    for leftover in range(len(usable), rows * 2):
        axes[leftover // 2][leftover % 2].axis("off")

    figure.suptitle(
        f"{result.ticker}: one-session-ahead forecasts on the held-out test window",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def plot_forecast(result, history, output: Path) -> Path:
    """Recent history plus each model's forward forecast."""
    figure, axis = plt.subplots(figsize=(11, 5))
    recent = history.frame.tail(60)
    axis.plot(recent.index, recent["adj_close"], label="Actual (adjusted close)", color=INK, linewidth=2.0)

    for index, report in enumerate(result.reports):
        if not report.ok or not report.forecast:
            continue
        dates = [pd.Timestamp(result.reference_date)] + [pd.Timestamp(s.target_date) for s in report.forecast]
        values = [result.reference_price] + [s.predicted_price for s in report.forecast]
        axis.plot(
            dates,
            values,
            marker="o",
            markersize=4,
            linewidth=2.0,
            linestyle="--",
            color=PALETTE[index % len(PALETTE)],
            label=report.display_name,
        )

    axis.set_title(
        f"{result.ticker}: {result.horizon}-session forecasts from {result.reference_date}",
        fontsize=12,
        fontweight="bold",
    )
    axis.set_ylabel("Price")
    axis.grid(alpha=GRID_ALPHA)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def leaderboard_markdown(rows: list[dict]) -> str:
    header = (
        "| Model | Family | Val MAE | Test MAE | Test RMSE | Test MAPE % | Dir. acc. | R² | "
        "MAE / naive | Beats naive | Fit (s) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    lines = []
    for row in rows:
        if row.get("error"):
            lines.append(
                f"| {row['model']} | {row.get('family', '-')} | "
                + " | ".join(["failed"] * 9)
                + " |"
            )
            continue
        lines.append(
            "| {model}{best} | {family} | {vmae} | {tmae} | {trmse} | {tmape} | {dir} | {r2} | "
            "{ratio} | {beats} | {secs} |".format(
                model=row["display_name"],
                best=" **(selected)**" if row["is_best"] else "",
                family=row["family"],
                vmae=_fmt(row["validation_mae"], 3),
                tmae=_fmt(row["test_mae"], 3),
                trmse=_fmt(row["test_rmse"], 3),
                tmape=_fmt(row["test_mape"], 2),
                dir=_fmt(row["test_directional_accuracy"], 3),
                r2=_fmt(row["test_r2"], 3),
                ratio=_fmt(row["mae_vs_naive_test"], 3),
                beats=_fmt(row["beats_naive_on_test"]),
                secs=_fmt(row["training_seconds"], 2),
            )
        )
    return header + "\n".join(lines) + "\n"


def run_one(ticker: str, horizon: int, models: list[str], history_days: int, output_dir: Path) -> dict | None:
    print(f"\n=== {ticker} ===", flush=True)
    try:
        history = load_history(ticker, days=history_days)
    except DataUnavailableError as exc:
        print(f"  skipped: {exc}", flush=True)
        return None

    print(
        f"  {len(history.frame)} sessions from {history.frame.index[0].date()} to "
        f"{history.last_date.date()} via {history.source}",
        flush=True,
    )
    try:
        result = run_experiment(history, horizon=horizon, model_names=models, history_days=history_days)
    except InsufficientHistoryError as exc:
        print(f"  skipped: {exc}", flush=True)
        return None

    rows = leaderboard_rows(result)
    safe = ticker.replace("^", "idx_").replace(".", "_")
    pd.DataFrame(rows).to_csv(output_dir / f"{safe}_leaderboard.csv", index=False)
    with open(output_dir / f"{safe}_experiment.json", "w", encoding="utf-8") as handle:
        json.dump(result.as_dict(include_curve=False), handle, indent=2)

    comparison_chart = plot_actual_vs_predicted(result, output_dir / f"{safe}_actual_vs_predicted.png")
    forecast_chart = plot_forecast(result, history, output_dir / f"{safe}_forecast.png")

    for row in rows:
        if row.get("error"):
            print(f"  {row['model']:<18} failed: {row['error']}", flush=True)
        else:
            print(
                f"  {row['model']:<18} val MAE {_fmt(row['validation_mae'], 3):>8}  "
                f"test MAE {_fmt(row['test_mae'], 3):>8}  vs naive {_fmt(row['mae_vs_naive_test'], 3):>6}",
                flush=True,
            )
    print(f"  selected by validation MAE: {result.best_model}", flush=True)

    return {
        "ticker": ticker,
        "result": result,
        "rows": rows,
        "charts": [p for p in (comparison_chart, forecast_chart) if p],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading sessions.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_SET)
    parser.add_argument("--history-days", type=int, default=1095)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    experiments = [
        outcome
        for ticker in args.tickers
        if (outcome := run_one(ticker, args.horizon, args.models, args.history_days, args.out))
    ]
    if not experiments:
        print("No experiment produced results.")
        return 1

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = [
        "# Experimental results",
        "",
        f"Generated by `scripts/run_experiments.py` on {generated}.",
        "",
        "Every number below was measured by the project's own pipeline. Reproduce with:",
        "",
        "```bash",
        f"python scripts/run_experiments.py --tickers {' '.join(args.tickers)} "
        f"--horizon {args.horizon} --models {' '.join(args.models)}",
        "```",
        "",
        "## Method",
        "",
        "* Prediction target: cumulative log return over the next *h* trading sessions, converted back to a price.",
        "* Features: lagged returns, moving-average ratios, RSI, MACD, realised volatility and volume features, "
        "each computed only from sessions up to the forecast origin.",
        "* Split: 70% train, 15% validation, 15% test, in chronological order with no shuffling.",
        "* Evaluation: walk-forward origins through each segment; cheap models refit on the expanding window, "
        "ARIMA and the recurrent models are fitted once per segment and rolled forward.",
        "* Selection: lowest one-session-ahead validation MAE. The test window is never used to choose a model.",
        "* `MAE / naive` below 1.0 means the model beat the naive last-value forecast on unseen data.",
        "",
    ]

    for item in experiments:
        result = item["result"]
        dataset = result.split.describe()
        sections += [
            f"## {result.ticker}" + (f" - {result.company_name}" if result.company_name else ""),
            "",
            f"* Data source: {result.data_source}, retrieved {result.data_retrieved_at:%Y-%m-%d %H:%M UTC}",
            f"* Sessions: {result.n_rows} downloaded, {result.feature_rows} usable after indicator warm-up",
            f"* Train {dataset['n_train']} ({result.split_dates['train_start']} to {result.split_dates['train_end']}), "
            f"validation {dataset['n_validation']} ({result.split_dates['validation_start']} to "
            f"{result.split_dates['validation_end']}), test {dataset['n_test']} "
            f"({result.split_dates['test_start']} to {result.split_dates['test_end']})",
            f"* Forecast origin: {result.reference_date} at {result.reference_price:.2f}",
            f"* Horizon: {result.horizon} trading session(s); total runtime {result.runtime_seconds:.1f}s",
            "",
            leaderboard_markdown(item["rows"]),
            "",
        ]
        beat = [r["display_name"] for r in item["rows"] if r.get("beats_naive_on_test")]
        sections += [
            f"Models that beat the naive baseline on the test window: "
            f"{', '.join(beat) if beat else 'none'}.",
            "",
            f"![Actual versus predicted]({result.ticker.replace('^', 'idx_').replace('.', '_')}_actual_vs_predicted.png)",
            "",
            f"![Forecast]({result.ticker.replace('^', 'idx_').replace('.', '_')}_forecast.png)",
            "",
        ]

    sections += [
        "## Reading these results",
        "",
        "A low MAE on a daily closing price is easy to achieve by predicting almost no change, which is exactly",
        "what the naive baseline does. The columns that matter are `MAE / naive` and directional accuracy: a model",
        "only demonstrates predictive skill when it beats the baseline on the test window, and directional accuracy",
        "near 0.5 is what an uninformative forecast produces on daily equity data.",
        "",
        "These results describe the tickers and periods listed above. They are not a general claim about the",
        "algorithms, and they are not investment advice.",
        "",
    ]

    summary = args.out / "RESULTS.md"
    summary.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nWrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
