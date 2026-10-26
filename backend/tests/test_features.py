"""Indicator correctness and the no-look-ahead guarantee (SRS 18, ML testing)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.features import (
    FEATURE_COLUMNS,
    add_indicators,
    build_feature_frame,
    build_supervised,
    build_targets,
    exponential_moving_average,
    macd,
    next_trading_days,
    prices_from_log_returns,
    relative_strength_index,
    simple_moving_average,
)


def test_simple_moving_average_matches_hand_calculation():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = simple_moving_average(series, 3)
    assert result.isna().tolist() == [True, True, False, False, False]
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_exponential_moving_average_recursion():
    series = pd.Series([10.0, 11.0, 12.0, 13.0])
    result = exponential_moving_average(series, 2)
    # span 2 -> alpha 2/3; seeded with the first value, then recursive.
    alpha = 2.0 / 3.0
    expected = 10.0
    for value in series.iloc[1:]:
        expected = alpha * value + (1 - alpha) * expected
    assert result.iloc[-1] == pytest.approx(expected)


def test_rsi_is_100_for_a_series_that_only_rises():
    series = pd.Series(np.arange(1.0, 40.0))
    rsi = relative_strength_index(series, 14)
    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_zero_for_a_series_that_only_falls():
    series = pd.Series(np.arange(40.0, 1.0, -1.0))
    rsi = relative_strength_index(series, 14)
    assert rsi.iloc[-1] == pytest.approx(0.0)


def test_rsi_stays_within_bounds(price_frame):
    rsi = relative_strength_index(price_frame["adj_close"], 14).dropna()
    assert rsi.between(0.0, 100.0).all()


def test_macd_is_the_difference_of_two_emas(price_frame):
    close = price_frame["adj_close"]
    macd_line, signal_line, histogram = macd(close)
    expected = exponential_moving_average(close, 12) - exponential_moving_average(close, 26)
    pd.testing.assert_series_equal(macd_line.dropna(), expected.dropna(), check_names=False)
    assert histogram.dropna().equals((macd_line - signal_line).dropna())


def test_targets_are_future_log_returns(price_frame):
    targets = build_targets(price_frame, horizon=3)
    close = price_frame["adj_close"]
    expected_one = np.log(close.iloc[1] / close.iloc[0])
    expected_three = np.log(close.iloc[3] / close.iloc[0])
    assert targets["y_1"].iloc[0] == pytest.approx(expected_one)
    assert targets["y_3"].iloc[0] == pytest.approx(expected_three)
    # The last rows have no known future and must be empty.
    assert targets["y_3"].tail(3).isna().all()


def test_features_do_not_change_when_future_rows_are_removed(price_frame):
    """The leakage guarantee: row t is a function of sessions up to t only."""
    cut = 400
    full = build_feature_frame(price_frame)
    truncated = build_feature_frame(price_frame.iloc[: cut + 1])
    shared = truncated.index.intersection(full.index)
    assert len(shared) > 200
    pd.testing.assert_frame_equal(
        full.loc[shared, FEATURE_COLUMNS], truncated.loc[shared, FEATURE_COLUMNS]
    )


def test_supervised_split_drops_rows_without_a_known_future(price_frame):
    features, targets = build_supervised(build_feature_frame(price_frame), horizon=5)
    assert len(features) == len(targets)
    assert targets.notna().all().all()
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.index.is_monotonic_increasing


def test_feature_frame_has_no_missing_or_infinite_values(feature_frame):
    values = feature_frame[FEATURE_COLUMNS]
    assert not values.isna().any().any()
    assert np.isfinite(values.to_numpy()).all()


def test_prices_from_log_returns_round_trips():
    prices = prices_from_log_returns(100.0, np.array([0.0, np.log(1.1)]))
    assert prices[0] == pytest.approx(100.0)
    assert prices[1] == pytest.approx(110.0)


def test_next_trading_days_skips_weekends():
    friday = pd.Timestamp("2026-10-16")
    days = next_trading_days(friday, 3)
    assert [d.strftime("%a") for d in days] == ["Mon", "Tue", "Wed"]


def test_indicator_columns_are_present(price_frame):
    enriched = add_indicators(price_frame)
    for column in ("sma_20", "ema_12", "rsi_14", "macd", "volatility_20", "volume_change"):
        assert column in enriched.columns
