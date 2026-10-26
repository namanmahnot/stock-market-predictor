"""Historical strategy backtest on walk-forward forecasts (SRS 10.5).

The simulation only ever acts on a forecast that was produced from sessions
before the traded session, charges a transaction cost whenever the position
changes, and is always reported next to a passive buy-and-hold benchmark.

A backtest measures what a decision rule would have done on one historical
window. It is not evidence that the rule will work, and it is reported as a
separate analytical module from forecast accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.data.providers import PriceHistory
from app.ml.pipeline import UNLIMITED_ORIGINS, prepare_dataset, walk_forward_evaluate
from app.ml.splits import chronological_split

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestMetrics:
    total_return_percent: float
    annualised_return_percent: float
    annualised_volatility_percent: float
    sharpe_ratio: float | None
    max_drawdown_percent: float
    final_equity: float

    def as_dict(self) -> dict[str, Any]:
        return {k: (None if v is None else round(float(v), 4)) for k, v in asdict(self).items()}


@dataclass
class BacktestResult:
    ticker: str
    model: str
    period_start: str
    period_end: str
    n_sessions: int
    threshold_percent: float
    transaction_cost_bps: float
    n_trades: int
    time_in_market_percent: float
    strategy: BacktestMetrics
    benchmark: BacktestMetrics
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "model": self.model,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "n_sessions": self.n_sessions,
            "threshold_percent": self.threshold_percent,
            "transaction_cost_bps": self.transaction_cost_bps,
            "n_trades": self.n_trades,
            "time_in_market_percent": round(self.time_in_market_percent, 2),
            "strategy": self.strategy.as_dict(),
            "benchmark": self.benchmark.as_dict(),
            "equity_curve": self.equity_curve,
            "notes": self.notes,
        }


def _summarise(returns: np.ndarray) -> BacktestMetrics:
    """Performance statistics for a series of simple per-session returns."""
    equity = np.cumprod(1.0 + returns) if returns.size else np.array([1.0])
    total = float(equity[-1] - 1.0)
    years = max(returns.size / TRADING_DAYS_PER_YEAR, 1e-9)
    annualised = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else -1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if returns.size > 1 else 0.0
    mean_daily = float(returns.mean()) if returns.size else 0.0
    sharpe = (
        float(mean_daily / returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if returns.size > 1 and returns.std(ddof=1) > 1e-12
        else None
    )
    peak = np.maximum.accumulate(equity)
    drawdown = float(((equity - peak) / peak).min()) if equity.size else 0.0
    return BacktestMetrics(
        total_return_percent=total * 100.0,
        annualised_return_percent=annualised * 100.0,
        annualised_volatility_percent=volatility * 100.0,
        sharpe_ratio=sharpe,
        max_drawdown_percent=drawdown * 100.0,
        final_equity=float(equity[-1]),
    )


def run_backtest(
    history: PriceHistory,
    model_name: str = "xgboost",
    threshold_percent: float = 0.0,
    transaction_cost_bps: float = 10.0,
    refit_every: int = 5,
    use_bulk_deals: bool | None = False,
    model_params: dict[str, Any] | None = None,
) -> BacktestResult:
    """Long/flat rule driven by the one-session-ahead forecast.

    The rule goes long for session ``t+1`` when the forecast made after the
    close of session ``t`` exceeds ``threshold_percent``, and otherwise stays in
    cash earning nothing.
    """
    prepared = prepare_dataset(history, use_bulk_deals=use_bulk_deals)
    frame = prepared.frame
    split = chronological_split(len(frame), settings.train_fraction, settings.validation_fraction)

    evaluation = walk_forward_evaluate(
        model_name,
        frame,
        split.test,
        horizon=1,
        segment_name="test",
        origin_step=1,
        refit_every=refit_every,
        max_origins=UNLIMITED_ORIGINS,
        model_params=model_params,
    )
    if not evaluation.curve:
        raise ValueError(
            f"No walk-forward forecasts were produced for {model_name}; the test window "
            "may be too short to backtest."
        )

    curve = pd.DataFrame(evaluation.curve)
    reference = curve["reference_price"].to_numpy(dtype="float64")
    actual = curve["actual"].to_numpy(dtype="float64")
    predicted = curve["predicted"].to_numpy(dtype="float64")

    realised = actual / reference - 1.0
    expected = predicted / reference - 1.0
    signal = (expected > threshold_percent / 100.0).astype("float64")

    cost = transaction_cost_bps / 10_000.0
    previous = np.concatenate([[0.0], signal[:-1]])
    turnover = np.abs(signal - previous)
    strategy_returns = signal * realised - turnover * cost

    strategy = _summarise(strategy_returns)
    benchmark = _summarise(realised)

    strategy_equity = np.cumprod(1.0 + strategy_returns)
    benchmark_equity = np.cumprod(1.0 + realised)
    equity_curve = [
        {
            "date": curve["target_date"].iloc[i],
            "strategy": round(float(strategy_equity[i]), 6),
            "benchmark": round(float(benchmark_equity[i]), 6),
            "in_market": bool(signal[i]),
        }
        for i in range(len(curve))
    ]

    notes = [
        *prepared.notes,
        "Positions are taken on the session after the forecast origin, so no future price is used.",
        f"A one-way cost of {transaction_cost_bps:.1f} basis points is charged whenever the position changes.",
        "Cash earns no interest, dividends beyond the adjusted close are ignored, and slippage is not modelled.",
        "Past simulated performance does not indicate future results.",
    ]
    if model_name == "naive":
        notes.append("The naive forecast never predicts a move, so this rule stays in cash throughout.")

    return BacktestResult(
        ticker=history.ticker,
        model=model_name,
        period_start=str(curve["target_date"].iloc[0]),
        period_end=str(curve["target_date"].iloc[-1]),
        n_sessions=len(curve),
        threshold_percent=threshold_percent,
        transaction_cost_bps=transaction_cost_bps,
        n_trades=int(turnover.sum()),
        time_in_market_percent=float(signal.mean() * 100.0),
        strategy=strategy,
        benchmark=benchmark,
        equity_curve=equity_curve,
        notes=notes,
    )
