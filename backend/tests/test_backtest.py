"""Strategy backtest: no future data, costs charged, benchmark reported."""
from __future__ import annotations

import pytest

from app.ml.backtest import run_backtest


@pytest.fixture(scope="module")
def backtest(request):
    from datetime import datetime, timezone

    from app.data.providers import PriceHistory
    from tests.conftest import make_price_frame

    history = PriceHistory("TEST", make_price_frame(), "csv", datetime.now(timezone.utc))
    return run_backtest(
        history, model_name="linear_regression", transaction_cost_bps=10.0, refit_every=10
    )


def test_backtest_covers_the_test_window_session_by_session(backtest):
    assert backtest.n_sessions > 50
    assert len(backtest.equity_curve) == backtest.n_sessions
    dates = [point["date"] for point in backtest.equity_curve]
    assert dates == sorted(dates)
    assert backtest.period_start == dates[0] and backtest.period_end == dates[-1]


def test_backtest_reports_a_passive_benchmark(backtest):
    assert backtest.benchmark.total_return_percent is not None
    assert backtest.strategy.final_equity > 0
    assert backtest.benchmark.final_equity > 0


def test_time_in_market_and_trade_count_are_consistent(backtest):
    assert 0.0 <= backtest.time_in_market_percent <= 100.0
    assert backtest.n_trades >= 0
    if backtest.time_in_market_percent == 0.0:
        assert backtest.n_trades == 0


def test_transaction_costs_reduce_the_result():
    from datetime import datetime, timezone

    from app.data.providers import PriceHistory
    from tests.conftest import make_price_frame

    history = PriceHistory("TEST", make_price_frame(), "csv", datetime.now(timezone.utc))
    free = run_backtest(history, "linear_regression", transaction_cost_bps=0.0, refit_every=10)
    costly = run_backtest(history, "linear_regression", transaction_cost_bps=50.0, refit_every=10)
    if free.n_trades > 0:
        assert costly.strategy.total_return_percent < free.strategy.total_return_percent


def test_a_flat_forecast_keeps_the_strategy_in_cash():
    from datetime import datetime, timezone

    from app.data.providers import PriceHistory
    from tests.conftest import make_price_frame

    history = PriceHistory("TEST", make_price_frame(), "csv", datetime.now(timezone.utc))
    result = run_backtest(history, "naive", refit_every=25)
    assert result.time_in_market_percent == 0.0
    assert result.strategy.total_return_percent == pytest.approx(0.0)
    assert any("stays in cash" in note for note in result.notes)


def test_notes_state_the_simulation_assumptions(backtest):
    joined = " ".join(backtest.notes)
    assert "no future price is used" in joined
    assert "basis points" in joined
    assert "does not indicate future results" in joined


def test_drawdown_is_never_positive(backtest):
    assert backtest.strategy.max_drawdown_percent <= 0.0
    assert backtest.benchmark.max_drawdown_percent <= 0.0


def test_payload_is_serialisable(backtest):
    import json

    payload = json.dumps(backtest.as_dict())
    assert "equity_curve" in payload and "benchmark" in payload
