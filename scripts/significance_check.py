"""Is any model's edge real, or is it noise?

Two modes, run separately and combined into docs/results/SIGNIFICANCE.md:

``exploratory``
    Every default model on AAPL, MSFT and RELIANCE.NS, scored at every session
    of the held-out test window. Tests each model's error against the naive
    forecast (Diebold-Mariano) and its directional accuracy against both 50%
    and an "always predict up" rule (exact McNemar). Because five models are
    tested, results are judged against a Bonferroni-corrected threshold.

``confirm``
    The pre-registered follow-up. The exploratory run flagged one candidate:
    the LSTM's directional accuracy. This mode tests that single hypothesis on
    stocks that were never used in any evaluation in this project. One
    hypothesis, one test, fixed before the data was seen: no correction is
    needed, and no other result from this mode is used as evidence.

Examples
--------
    python scripts/significance_check.py exploratory
    python scripts/significance_check.py confirm
    python scripts/significance_check.py report      # rebuild the Markdown only
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.data.providers import DataUnavailableError, load_history  # noqa: E402
from app.ml.pipeline import (  # noqa: E402
    UNLIMITED_ORIGINS,
    InsufficientHistoryError,
    prepare_dataset,
    walk_forward_evaluate,
)
from app.ml.registry import DEFAULT_MODEL_SET  # noqa: E402
from app.ml.splits import chronological_split  # noqa: E402

RESULTS = PROJECT_ROOT / "docs" / "results"
ALPHA = 0.05
HISTORY_DAYS = 1095

EXPLORATORY_TICKERS = ["AAPL", "MSFT", "RELIANCE.NS"]
EXPLORATORY_MODELS = [m for m in DEFAULT_MODEL_SET if m != "naive"]

# ---------------------------------------------------------------------------
# Pre-registration. Fixed before the confirmatory run; do not edit after it.
# ---------------------------------------------------------------------------
CONFIRM_MODEL = "lstm"
CONFIRM_HYPOTHESIS = (
    "On stocks never used in any evaluation in this project, the LSTM's one-session-ahead "
    "directional accuracy is higher than that of always predicting 'up', on the same sessions."
)
#: Every stock in the application's search universe that no evaluation in this
#: project has touched. Excluded: AAPL, MSFT, RELIANCE.NS (exploratory run),
#: TCS.NS (live API checks), the four bulk-deal small caps (ablation), SPY and
#: ^NSEI (an ETF and an index rather than stocks).
CONFIRM_TICKERS = [
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "TATAMOTORS.NS",
    "WIPRO.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
]


def score(ticker: str, model: str) -> dict | None:
    """Walk the test window one session at a time and summarise the forecasts."""
    try:
        history = load_history(ticker, days=HISTORY_DAYS)
        frame = prepare_dataset(history, use_bulk_deals=False).frame
    except (DataUnavailableError, InsufficientHistoryError) as exc:
        print(f"  {ticker}: skipped ({exc})", flush=True)
        return None

    split = chronological_split(len(frame), settings.train_fraction, settings.validation_fraction)
    evaluation = walk_forward_evaluate(
        model,
        frame,
        split.test,
        horizon=1,
        segment_name="test",
        origin_step=1,
        refit_every=5,
        max_origins=UNLIMITED_ORIGINS,
    )
    curve = pd.DataFrame(evaluation.curve)
    if curve.empty:
        print(f"  {ticker}: no forecasts produced", flush=True)
        return None

    actual = curve["actual"].to_numpy()
    predicted = curve["predicted"].to_numpy()
    reference = curve["reference_price"].to_numpy()

    model_abs = np.abs(actual - predicted)
    naive_abs = np.abs(actual - reference)  # the naive forecast is the reference price
    # Diebold-Mariano for h = 1: one-sided t-test on the loss differential.
    dm_p = float(stats.ttest_1samp(model_abs - naive_abs, 0.0, alternative="less").pvalue)

    actual_move = np.sign(actual - reference)
    predicted_move = np.sign(predicted - reference)
    usable = (actual_move != 0) & (predicted_move != 0)
    model_right = actual_move[usable] == predicted_move[usable]
    up_right = actual_move[usable] > 0
    hits = int(model_right.sum())
    n = int(usable.sum())

    # Exact McNemar: sessions where exactly one of the two rules was right.
    model_only = int((model_right & ~up_right).sum())
    up_only = int((~model_right & up_right).sum())
    discordant = model_only + up_only
    mcnemar_p = (
        float(stats.binomtest(model_only, discordant, 0.5, alternative="greater").pvalue)
        if discordant
        else 1.0
    )

    return {
        "ticker": ticker,
        "model": model,
        "test_start": str(curve["target_date"].iloc[0]),
        "test_end": str(curve["target_date"].iloc[-1]),
        "forecasts": len(curve),
        "mae": float(model_abs.mean()),
        "naive_mae": float(naive_abs.mean()),
        "skill": float(model_abs.mean() / naive_abs.mean()),
        "mape": float(np.mean(model_abs / actual) * 100),
        "dm_p": dm_p,
        "dir_n": n,
        "hits": hits,
        "ups": int(up_right.sum()),
        "model_only_right": model_only,
        "up_only_right": up_only,
        "dir_p_vs_half": float(stats.binomtest(hits, n, 0.5, alternative="greater").pvalue) if n else None,
        "mcnemar_p": mcnemar_p,
    }


def pooled(frame: pd.DataFrame, model: str) -> dict:
    sub = frame[frame["model"] == model]
    hits, n, ups = int(sub["hits"].sum()), int(sub["dir_n"].sum()), int(sub["ups"].sum())
    model_only, up_only = int(sub["model_only_right"].sum()), int(sub["up_only_right"].sum())
    discordant = model_only + up_only
    ci = stats.binomtest(hits, n, 0.5).proportion_ci(confidence_level=0.95)
    return {
        "model": model,
        "tickers": int(sub["ticker"].nunique()),
        "hits": hits,
        "n": n,
        "accuracy": hits / n,
        "ci_low": ci.low,
        "ci_high": ci.high,
        "always_up": ups / n,
        "model_only_right": model_only,
        "up_only_right": up_only,
        "p_vs_half": float(stats.binomtest(hits, n, 0.5, alternative="greater").pvalue),
        "mcnemar_p": float(stats.binomtest(model_only, discordant, 0.5, alternative="greater").pvalue)
        if discordant
        else 1.0,
    }


def run(mode: str) -> Path:
    tickers = EXPLORATORY_TICKERS if mode == "exploratory" else CONFIRM_TICKERS
    models = EXPLORATORY_MODELS if mode == "exploratory" else [CONFIRM_MODEL]
    rows = []
    for ticker in tickers:
        for model in models:
            print(f"{mode}: {ticker} / {model}", flush=True)
            result = score(ticker, model)
            if result:
                rows.append(result)
    out = RESULTS / f"significance_{mode}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}", flush=True)
    return out


def _p(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def report() -> Path:
    lines = [
        "# Statistical significance of the forecasting results",
        "",
        f"Generated by `scripts/significance_check.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}.",
        "",
        "The other result files report how large each model's error was. This one asks whether any",
        "difference from the naive forecast is **larger than chance**. Every model here is scored at",
        "*every* session of the held-out test window, one session ahead, refitting cheap models every",
        "five sessions on the expanding window; ARIMA and the LSTM are fitted once per window.",
        "",
        "Tests used:",
        "",
        "* **Error versus naive** — Diebold-Mariano test (one-sided) on the per-session difference in",
        "  absolute error. A small p-value means the model's error is reliably lower than the naive",
        "  last-value forecast's.",
        "* **Direction versus 50%** — one-sided binomial test.",
        "* **Direction versus always-up** — exact McNemar test on the same sessions. This matters",
        "  because in a rising market, always predicting 'up' beats 50% without any skill.",
        "",
    ]

    exploratory_path = RESULTS / "significance_exploratory.csv"
    if exploratory_path.exists():
        frame = pd.read_csv(exploratory_path)
        n_models = frame["model"].nunique()
        threshold = ALPHA / n_models
        lines += [
            "## Part 1 — Exploratory: all models, three stocks",
            "",
            f"Five models were tested, so a result needs p < {ALPHA}/{n_models} = **{threshold:.3f}**",
            "(Bonferroni) before it counts. Anything between that and 0.05 is a lead, not a finding.",
            "",
            "| Stock | Model | Forecasts | MAPE | Error ÷ naive | p (error < naive) | Direction | p (> always-up) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in frame.sort_values(["ticker", "skill"]).itertuples():
            lines.append(
                f"| {row.ticker} | {row.model} | {row.forecasts} | {row.mape:.2f}% | {row.skill:.3f} | "
                f"{_p(row.dm_p)} | {row.hits / row.dir_n:.3f} | {_p(row.mcnemar_p)} |"
            )

        lines += [
            "",
            "**Pooled directional accuracy across the three stocks**",
            "",
            "| Model | Correct | Accuracy | 95% CI | Always-up | p (> 50%) | p (> always-up) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        pooled_rows = [pooled(frame, model) for model in sorted(frame["model"].unique())]
        for p in pooled_rows:
            lines.append(
                f"| {p['model']} | {p['hits']}/{p['n']} | {p['accuracy']:.3f} | "
                f"[{p['ci_low']:.3f}, {p['ci_high']:.3f}] | {p['always_up']:.3f} | "
                f"{_p(p['p_vs_half'])} | {_p(p['mcnemar_p'])} |"
            )

        error_wins = frame[frame["dm_p"] < threshold]
        direction_wins = [p for p in pooled_rows if p["mcnemar_p"] < threshold]
        # A lead is nominally significant on either directional test but does
        # not survive the correction; that is what earns a confirmation run.
        leads = [
            p
            for p in pooled_rows
            if p not in direction_wins and min(p["p_vs_half"], p["mcnemar_p"]) < ALPHA
        ]
        lines += [
            "",
            "**Reading Part 1.** "
            + (
                "No model's error was reliably lower than the naive forecast's on any stock. "
                if error_wins.empty
                else f"{len(error_wins)} model/stock pair(s) beat the naive forecast's error after correction. "
            )
            + (
                "No model's directional accuracy beat always-up after correction. "
                if not direction_wins
                else f"Directional accuracy beat always-up after correction for: "
                f"{', '.join(p['model'] for p in direction_wins)}. "
            )
            + (
                "Lead(s) nominally significant before correction, carried forward to Part 2: "
                + ", ".join(
                    f"{p['model']} ({p['accuracy']:.3f}; p = {_p(p['p_vs_half'])} vs 50%, "
                    f"{_p(p['mcnemar_p'])} vs always-up)"
                    for p in leads
                )
                + "."
                if leads
                else ""
            ),
            "",
        ]

    confirm_path = RESULTS / "significance_confirm.csv"
    lines += [
        "## Part 2 — Confirmatory: one pre-registered hypothesis, fresh stocks",
        "",
        "A lead found by testing many things on one dataset has to be checked on data it has never",
        "seen, with the question fixed in advance. The hypothesis, the model and the stock list below",
        "were written into the script before this run, and no other comparison from it is used as evidence.",
        "",
        f"> **Hypothesis.** {CONFIRM_HYPOTHESIS}",
        ">",
        f"> **Model.** `{CONFIRM_MODEL}`, unchanged configuration and seed.",
        ">",
        f"> **Stocks.** {', '.join(CONFIRM_TICKERS)} — every stock in the application's search list",
        "> that no evaluation in this project had used.",
        ">",
        f"> **Test.** Exact McNemar, one-sided, pooled across stocks, α = {ALPHA}. One test, so no correction.",
        "",
    ]
    if confirm_path.exists():
        confirm = pd.read_csv(confirm_path)
        skipped = sorted(set(CONFIRM_TICKERS) - set(confirm["ticker"]))
        lines += [
            "| Stock | Test window | Sessions | LSTM direction | Always-up | LSTM-only right | Up-only right |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in confirm.itertuples():
            lines.append(
                f"| {row.ticker} | {row.test_start} → {row.test_end} | {row.dir_n} | "
                f"{row.hits / row.dir_n:.3f} | {row.ups / row.dir_n:.3f} | "
                f"{row.model_only_right} | {row.up_only_right} |"
            )
        p = pooled(confirm, CONFIRM_MODEL)
        confirmed = p["mcnemar_p"] < ALPHA
        lines += [
            "",
            f"**Pooled over {p['tickers']} stocks:** LSTM {p['hits']}/{p['n']} = **{p['accuracy']:.3f}** "
            f"(95% CI [{p['ci_low']:.3f}, {p['ci_high']:.3f}]) versus always-up **{p['always_up']:.3f}**. "
            f"On the sessions where they disagreed, the LSTM alone was right {p['model_only_right']} times "
            f"and always-up alone {p['up_only_right']} times.",
            "",
            f"**McNemar p = {_p(p['mcnemar_p'])}.**",
            "",
        ]
        if skipped:
            lines += [
                f"Stocks skipped because the provider returned no usable history: {', '.join(skipped)}. "
                "They are listed rather than replaced, so the stock list stays as pre-registered.",
                "",
            ]
        lines += [
            "### Verdict",
            "",
            (
                "**Confirmed.** On fresh stocks the LSTM predicted direction better than always-up by more "
                "than chance. That is a real, if modest, directional signal, and it should be reported "
                "together with the fact that its price error did not beat the naive forecast."
                if confirmed
                else "**Not confirmed.** On fresh stocks the LSTM's directional accuracy was not "
                "distinguishable from always predicting 'up'. The exploratory result was most likely the "
                "product of testing five models on three stocks: with that many comparisons, one of them "
                "clearing p ≈ 0.01 by chance is unremarkable. This is the correct, expected outcome of a "
                "confirmation test, and reporting it is worth more than the original lead."
            ),
            "",
        ]
    else:
        lines += ["*Not run yet: `python scripts/significance_check.py confirm`.*", ""]

    lines += [
        "## Conclusion for the report",
        "",
        "No model produced a reliably lower price error than repeating the previous close. Any",
        "directional claim must rest on Part 2, which was designed so that a positive result could",
        "not be manufactured by searching.",
        "",
    ]

    out = RESULTS / "SIGNIFICANCE.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["exploratory", "confirm", "report"])
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.mode != "report":
        run(args.mode)
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
