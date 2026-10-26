"""Provider validation: bad input, dirty data and provider outages (SRS 18)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.providers import (
    DataUnavailableError,
    InvalidTickerError,
    _tidy,
    load_history,
    normalise_ticker,
    search_symbols,
)
from tests.conftest import SYNTHETIC_TICKER, make_price_frame


def test_normalise_ticker_accepts_common_formats():
    assert normalise_ticker(" aapl ") == "AAPL"
    assert normalise_ticker("reliance.ns") == "RELIANCE.NS"
    assert normalise_ticker("^nsei") == "^NSEI"
    assert normalise_ticker("brk-b") == "BRK-B"


@pytest.mark.parametrize("value", ["", "   ", "AA PL", "DROP TABLE users", "a" * 40, "aapl;--"])
def test_normalise_ticker_rejects_malformed_symbols(value):
    with pytest.raises(InvalidTickerError):
        normalise_ticker(value)


def test_tidy_removes_duplicate_trading_dates():
    frame = make_price_frame(30)
    duplicated = pd.concat([frame, frame.iloc[[-1]]])
    tidy, warnings = _tidy(duplicated, "TEST")
    assert len(tidy) == 30
    assert not tidy.index.duplicated().any()
    assert any("duplicate" in warning for warning in warnings)


def test_tidy_drops_rows_with_missing_prices():
    frame = make_price_frame(30)
    frame.loc[frame.index[-1], ["close", "adj_close"]] = np.nan
    tidy, warnings = _tidy(frame, "TEST")
    assert len(tidy) == 29
    assert any("missing or non-positive" in warning for warning in warnings)


def test_tidy_drops_non_positive_prices():
    frame = make_price_frame(30)
    frame.loc[frame.index[5], "low"] = -1.0
    tidy, _ = _tidy(frame, "TEST")
    assert len(tidy) == 29


def test_tidy_sorts_sessions_chronologically():
    frame = make_price_frame(30)
    tidy, _ = _tidy(frame.iloc[::-1], "TEST")
    assert tidy.index.is_monotonic_increasing


def test_tidy_reports_a_missing_column():
    frame = make_price_frame(30).drop(columns=["volume"])
    with pytest.raises(DataUnavailableError, match="missing columns"):
        _tidy(frame, "TEST")


def test_tidy_falls_back_to_the_raw_close_when_no_adjusted_close_exists():
    frame = make_price_frame(30).drop(columns=["adj_close"])
    tidy, warnings = _tidy(frame, "TEST")
    assert tidy["adj_close"].equals(tidy["close"])
    assert any("adjusted close" in warning for warning in warnings)


def test_load_history_reads_the_offline_csv(offline_provider):
    history = load_history(SYNTHETIC_TICKER, days=3650)
    assert history.source == "csv"
    assert len(history.frame) > 700
    assert history.frame.index.is_monotonic_increasing
    assert history.last_close > 0


def test_load_history_fails_loudly_for_an_unknown_symbol(offline_provider):
    """A provider with no data must raise, never return invented prices."""
    with pytest.raises(DataUnavailableError):
        load_history("NOSUCHTICKER", days=365, use_cache=False)


def test_load_history_rejects_an_invalid_symbol(offline_provider):
    with pytest.raises(InvalidTickerError):
        load_history("not a ticker", days=365)


def test_search_matches_ticker_and_company_name():
    assert any(entry["ticker"] == "AAPL" for entry in search_symbols("appl"))
    assert any(entry["ticker"] == "RELIANCE.NS" for entry in search_symbols("reliance"))
    assert any(entry["ticker"] == "TCS.NS" for entry in search_symbols("tcs"))


def test_search_passes_through_an_unlisted_but_valid_symbol():
    results = search_symbols("ZZZZ")
    assert results and results[0]["ticker"] == "ZZZZ"
    assert results[0]["name"] == "Unverified symbol"


def test_search_with_an_empty_query_returns_the_universe():
    assert len(search_symbols("")) > 0
