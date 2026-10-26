"""Market data providers behind one common interface (SRS 5.1, 11.1).

The rest of the application only ever sees :class:`PriceHistory`, so a provider
can be replaced without touching the API contract or the frontend.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.data import cache

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
TICKER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,19}$")

#: Small curated universe so stock search works without a paid symbol endpoint.
SYMBOL_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "currency": "USD"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "currency": "USD"},
    {"ticker": "GOOGL", "name": "Alphabet Inc. Class A", "exchange": "NASDAQ", "currency": "USD"},
    {"ticker": "AMZN", "name": "Amazon.com, Inc.", "exchange": "NASDAQ", "currency": "USD"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "currency": "USD"},
    {"ticker": "META", "name": "Meta Platforms, Inc.", "exchange": "NASDAQ", "currency": "USD"},
    {"ticker": "TSLA", "name": "Tesla, Inc.", "exchange": "NASDAQ", "currency": "USD"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE", "currency": "USD"},
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE ARCA", "currency": "USD"},
    {"ticker": "^NSEI", "name": "NIFTY 50 Index", "exchange": "NSE", "currency": "INR"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "INFY.NS", "name": "Infosys Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "SBIN.NS", "name": "State Bank of India", "exchange": "NSE", "currency": "INR"},
    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "WIPRO.NS", "name": "Wipro Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "ITC.NS", "name": "ITC Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel Limited", "exchange": "NSE", "currency": "INR"},
    # Small and mid caps that appear in the bundled NSE bulk-deal export, so the
    # disclosure features can be demonstrated from the search box.
    {"ticker": "ATALREAL.NS", "name": "Atal Realtech Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "RATNAVEER.NS", "name": "Ratnaveer Precision Engineering Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "AGIIL.NS", "name": "AGI Infra Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "GCSL.NS", "name": "Gretex Corporate Services Limited", "exchange": "NSE", "currency": "INR"},
    {"ticker": "MOTISONS.NS", "name": "Motisons Jewellers Limited", "exchange": "NSE", "currency": "INR"},
]


class DataUnavailableError(RuntimeError):
    """Raised when no trustworthy history can be produced for a ticker.

    The application never fabricates prices to fill a provider outage.
    """


class InvalidTickerError(ValueError):
    pass


@dataclass
class PriceHistory:
    """Validated OHLCV history plus the provenance shown in the UI."""

    ticker: str
    frame: pd.DataFrame
    source: str
    retrieved_at: datetime
    name: str | None = None
    currency: str | None = None
    warnings: list[str] = field(default_factory=list)
    from_cache: bool = False

    @property
    def last_date(self) -> pd.Timestamp:
        return self.frame.index[-1]

    @property
    def last_close(self) -> float:
        return float(self.frame["adj_close"].iloc[-1])


def normalise_ticker(raw: str) -> str:
    ticker = (raw or "").strip().upper()
    if not ticker:
        raise InvalidTickerError("A ticker symbol is required.")
    probe = ticker[1:] if ticker.startswith("^") else ticker
    if not TICKER_RE.match(probe):
        raise InvalidTickerError(
            f"'{raw}' is not a valid ticker symbol. Use letters, digits, '.' or '-'."
        )
    return ticker


def _tidy(frame: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, list[str]]:
    """Normalise column names, drop unusable rows and report what was removed."""
    warnings: list[str] = []
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.droplevel(-1, axis=1)
    frame = frame.rename(columns={str(c): str(c).strip().lower().replace(" ", "_") for c in frame.columns})
    if "adj_close" not in frame.columns and "close" in frame.columns:
        frame["adj_close"] = frame["close"]
        warnings.append("Provider returned no adjusted close; using the raw close.")
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise DataUnavailableError(f"Provider response for {ticker} is missing columns: {missing}.")

    frame = frame[OHLCV_COLUMNS].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    frame = frame.apply(pd.to_numeric, errors="coerce")

    duplicates = int(frame.index.duplicated().sum())
    if duplicates:
        frame = frame[~frame.index.duplicated(keep="last")]
        warnings.append(f"Removed {duplicates} duplicate trading date(s).")

    frame = frame.sort_index()

    invalid = frame[["open", "high", "low", "close", "adj_close"]].le(0).any(axis=1)
    unusable = frame["adj_close"].isna() | frame["close"].isna() | invalid
    if int(unusable.sum()):
        # An in-progress session is reported with NaN prices; it is not a forecast input.
        warnings.append(f"Dropped {int(unusable.sum())} row(s) with missing or non-positive prices.")
        frame = frame[~unusable]

    frame["volume"] = frame["volume"].fillna(0.0)
    if frame.empty:
        raise DataUnavailableError(f"No usable price rows were returned for {ticker}.")
    return frame, warnings


class YFinanceProvider:
    """Yahoo Finance history through the yfinance library (student prototype)."""

    name = "yfinance"

    def fetch(self, ticker: str, days: int) -> tuple[pd.DataFrame, str | None, str | None]:
        import yfinance as yf

        start = (datetime.now(timezone.utc) - timedelta(days=int(days * 1.15) + 10)).date()
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise DataUnavailableError(
                f"Yahoo Finance returned no history for '{ticker}'. "
                "The symbol may be unsupported, delisted or rate limited."
            )
        return raw, self._company_name(ticker), self._currency(ticker)

    @staticmethod
    def _company_name(ticker: str) -> str | None:
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).get_info()
            return info.get("shortName") or info.get("longName")
        except Exception:  # metadata is optional; never fail a forecast over it
            logger.debug("Could not read company metadata for %s", ticker, exc_info=True)
            return None

    @staticmethod
    def _currency(ticker: str) -> str | None:
        for entry in SYMBOL_UNIVERSE:
            if entry["ticker"] == ticker:
                return entry["currency"]
        return "INR" if ticker.endswith((".NS", ".BO")) else "USD"


class CsvProvider:
    """Offline provider reading data/offline/<TICKER>.csv.

    Used for lab demonstrations without internet access and by the test suite.
    """

    name = "csv"

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory or settings.offline_csv_dir)

    def fetch(self, ticker: str, days: int) -> tuple[pd.DataFrame, str | None, str | None]:
        path = self.directory / f"{ticker}.csv"
        if not path.exists():
            raise DataUnavailableError(
                f"No offline CSV for '{ticker}'. Expected {path}."
            )
        frame = pd.read_csv(path)
        date_col = next((c for c in frame.columns if c.strip().lower() in {"date", "datetime"}), None)
        if date_col is None:
            raise DataUnavailableError(f"{path} has no date column.")
        frame = frame.set_index(date_col)
        return frame, None, None


def get_provider(name: str | None = None):
    provider = (name or settings.data_provider).lower()
    if provider == "csv":
        return CsvProvider()
    if provider == "yfinance":
        return YFinanceProvider()
    raise ValueError(f"Unknown data provider '{provider}'.")


def load_history(
    ticker: str,
    days: int | None = None,
    provider_name: str | None = None,
    use_cache: bool = True,
) -> PriceHistory:
    """Return validated history for a ticker, preferring a fresh cache entry.

    When the provider fails, an expired cache entry is served with a warning
    rather than inventing prices (acceptance test: no fabricated history).
    """
    ticker = normalise_ticker(ticker)
    days = min(int(days or settings.default_history_days), settings.max_history_days)
    provider = get_provider(provider_name)
    key = f"{provider.name}_{ticker}_{days}"

    if use_cache:
        cached = cache.read(key)
        if cached is not None:
            frame, retrieved_at = cached
            tidy, warnings = _tidy(frame, ticker)
            return PriceHistory(
                ticker=ticker,
                frame=tidy,
                source=provider.name,
                retrieved_at=retrieved_at,
                currency=YFinanceProvider._currency(ticker),
                warnings=warnings,
                from_cache=True,
            )

    try:
        raw, name, currency = provider.fetch(ticker, days)
    except DataUnavailableError:
        fallback = cache.stale(key) if use_cache else None
        if fallback is None:
            raise
        frame, retrieved_at = fallback
        tidy, warnings = _tidy(frame, ticker)
        warnings.append("Live provider request failed; showing the last cached download.")
        return PriceHistory(
            ticker=ticker,
            frame=tidy,
            source=f"{provider.name} (cached)",
            retrieved_at=retrieved_at,
            warnings=warnings,
            from_cache=True,
        )

    tidy, warnings = _tidy(raw, ticker)
    cutoff = tidy.index.max() - pd.Timedelta(days=days)
    tidy = tidy[tidy.index >= cutoff]
    if use_cache:
        cache.write(key, tidy)

    gaps = _calendar_gaps(tidy)
    if gaps:
        warnings.append(f"{gaps} gap(s) longer than four calendar days in the trading history.")

    return PriceHistory(
        ticker=ticker,
        frame=tidy,
        source=provider.name,
        retrieved_at=datetime.now(timezone.utc),
        name=name,
        currency=currency,
        warnings=warnings,
    )


def _calendar_gaps(frame: pd.DataFrame) -> int:
    if len(frame) < 2:
        return 0
    deltas = pd.Series(frame.index).diff().dropna()
    return int((deltas > pd.Timedelta(days=4)).sum())


def search_symbols(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Substring search over the bundled universe (FR-01)."""
    q = (query or "").strip().lower()
    if not q:
        return SYMBOL_UNIVERSE[:limit]
    matches = [
        entry
        for entry in SYMBOL_UNIVERSE
        if q in entry["ticker"].lower() or q in entry["name"].lower()
    ]
    if not matches and TICKER_RE.match(q.lstrip("^")):
        # Allow any symbol the provider might support, flagged as unverified.
        matches = [{"ticker": q.upper(), "name": "Unverified symbol", "exchange": "-", "currency": "-"}]
    return matches[:limit]
