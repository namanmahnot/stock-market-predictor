"""NSE bulk-deal disclosures as a modelling input.

A bulk deal is a single client trading more than 0.5% of a company's listed
equity in one session. NSE publishes the list **after the close** of that
session, so a forecast made after the close of session *t* may legitimately use
the deals disclosed for session *t* - the same availability rule the price
features follow. ``settings.bulk_deals_publication_lag`` can delay that by one
or more sessions for a stricter assumption.

Two things about this dataset shape the design:

* It is **sparse**. Most symbols appear on a handful of sessions, and large-cap
  stocks essentially never appear, because moving 0.5% of their equity in one
  deal is impractical. Coverage is therefore measured per ticker and reported
  before the features are used.
* It covers a **fixed window**. Outside that window "no deal recorded" is
  indistinguishable from "no data", so the dataset is clipped to the covered
  period rather than padded with zeros, which would otherwise teach a model
  that bulk deals never happened before the file starts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

#: Columns as exported by NSE, before whitespace is stripped.
REQUIRED_COLUMNS = {
    "Date",
    "Symbol",
    "Security Name",
    "Client Name",
    "Buy / Sell",
    "Quantity Traded",
    "Trade Price / Wght. Avg. Price",
}

COLUMN_RENAMES = {
    "Date": "deal_date",
    "Symbol": "symbol",
    "Security Name": "security_name",
    "Client Name": "client_name",
    "Buy / Sell": "side",
    "Quantity Traded": "quantity",
    "Trade Price / Wght. Avg. Price": "price",
    "Remarks": "remarks",
}

#: Daily aggregate columns produced for one ticker.
DAILY_COLUMNS = [
    "deal_count",
    "buy_quantity",
    "sell_quantity",
    "net_quantity",
    "gross_quantity",
    "buy_turnover",
    "sell_turnover",
    "gross_turnover",
    "deal_vwap",
    "buyer_count",
    "seller_count",
]


class BulkDealsUnavailableError(RuntimeError):
    """Raised when bulk-deal features are requested but cannot be supplied."""


@dataclass(frozen=True)
class Coverage:
    """How much of a ticker's history this dataset can actually describe."""

    ticker: str
    symbol: str | None
    in_dataset: bool
    sessions_with_deals: int
    deal_rows: int
    first_deal: date | None
    last_deal: date | None
    dataset_start: date
    dataset_end: date

    @property
    def sufficient(self) -> bool:
        """Whether there are enough disclosed sessions to be worth learning from."""
        return self.sessions_with_deals >= settings.bulk_deals_min_sessions

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "symbol": self.symbol,
            "in_dataset": self.in_dataset,
            "sessions_with_deals": self.sessions_with_deals,
            "deal_rows": self.deal_rows,
            "first_deal": self.first_deal.isoformat() if self.first_deal else None,
            "last_deal": self.last_deal.isoformat() if self.last_deal else None,
            "dataset_start": self.dataset_start.isoformat(),
            "dataset_end": self.dataset_end.isoformat(),
            "sufficient": self.sufficient,
            "minimum_sessions_required": settings.bulk_deals_min_sessions,
        }


@dataclass(frozen=True)
class BulkDealsDataset:
    """Every parsed disclosure, indexed for per-ticker lookup."""

    deals: pd.DataFrame
    sources: tuple[str, ...]

    @property
    def start(self) -> date:
        return self.deals["deal_date"].min().date()

    @property
    def end(self) -> date:
        return self.deals["deal_date"].max().date()

    @property
    def symbols(self) -> list[str]:
        return sorted(self.deals["symbol"].unique().tolist())

    def for_ticker(self, ticker: str) -> pd.DataFrame:
        """Raw disclosure rows for one ticker, oldest first."""
        symbol = symbol_for_ticker(ticker)
        if symbol is None:
            return self.deals.iloc[0:0]
        return self.deals[self.deals["symbol"] == symbol].sort_values(["deal_date", "client_name"])

    def coverage(self, ticker: str) -> Coverage:
        symbol = symbol_for_ticker(ticker)
        rows = self.for_ticker(ticker)
        return Coverage(
            ticker=ticker.upper(),
            symbol=symbol,
            in_dataset=not rows.empty,
            sessions_with_deals=int(rows["deal_date"].nunique()) if not rows.empty else 0,
            deal_rows=int(len(rows)),
            first_deal=rows["deal_date"].min().date() if not rows.empty else None,
            last_deal=rows["deal_date"].max().date() if not rows.empty else None,
            dataset_start=self.start,
            dataset_end=self.end,
        )

    def daily(self, ticker: str) -> pd.DataFrame:
        """One row per session on which this ticker had a disclosed deal."""
        rows = self.for_ticker(ticker)
        if rows.empty:
            return pd.DataFrame(columns=DAILY_COLUMNS, index=pd.DatetimeIndex([], name="date"))

        buys = rows[rows["side"] == "BUY"]
        sells = rows[rows["side"] == "SELL"]
        grouped = pd.DataFrame(
            {
                "deal_count": rows.groupby("deal_date").size(),
                "buy_quantity": buys.groupby("deal_date")["quantity"].sum(),
                "sell_quantity": sells.groupby("deal_date")["quantity"].sum(),
                "buy_turnover": buys.groupby("deal_date")["turnover"].sum(),
                "sell_turnover": sells.groupby("deal_date")["turnover"].sum(),
                "buyer_count": buys.groupby("deal_date")["client_name"].nunique(),
                "seller_count": sells.groupby("deal_date")["client_name"].nunique(),
                "gross_turnover": rows.groupby("deal_date")["turnover"].sum(),
                "gross_quantity": rows.groupby("deal_date")["quantity"].sum(),
            }
        ).fillna(0.0)

        grouped["net_quantity"] = grouped["buy_quantity"] - grouped["sell_quantity"]
        # Quantity-weighted average price across the session's disclosed deals.
        grouped["deal_vwap"] = (grouped["gross_turnover"] / grouped["gross_quantity"]).where(
            grouped["gross_quantity"] > 0
        )
        grouped.index = pd.DatetimeIndex(grouped.index, name="date")
        return grouped[DAILY_COLUMNS].sort_index()

    def coverage_table(self, limit: int | None = None) -> list[dict]:
        """Symbols ranked by the number of sessions they appear on."""
        grouped = (
            self.deals.groupby("symbol")
            .agg(
                deal_rows=("deal_date", "size"),
                sessions_with_deals=("deal_date", "nunique"),
                first_deal=("deal_date", "min"),
                last_deal=("deal_date", "max"),
                gross_turnover=("turnover", "sum"),
                security_name=("security_name", "first"),
            )
            .sort_values(["sessions_with_deals", "deal_rows"], ascending=False)
        )
        if limit:
            grouped = grouped.head(limit)
        return [
            {
                "symbol": symbol,
                "ticker": ticker_for_symbol(symbol),
                "security_name": row.security_name,
                "deal_rows": int(row.deal_rows),
                "sessions_with_deals": int(row.sessions_with_deals),
                "first_deal": row.first_deal.date().isoformat(),
                "last_deal": row.last_deal.date().isoformat(),
                "gross_turnover": float(row.gross_turnover),
                "sufficient": int(row.sessions_with_deals) >= settings.bulk_deals_min_sessions,
            }
            for symbol, row in grouped.iterrows()
        ]


def symbol_for_ticker(ticker: str) -> str | None:
    """Map a Yahoo Finance ticker to its NSE symbol.

    The dataset is an NSE report, so only ``.NS`` tickers (or bare NSE symbols)
    can be matched; anything else returns None.
    """
    value = (ticker or "").strip().upper()
    if not value or value.startswith("^"):
        return None
    if value.endswith(".NS"):
        return value[:-3]
    if "." in value:  # another exchange suffix, e.g. .BO
        return None
    return value


def ticker_for_symbol(symbol: str) -> str:
    return f"{symbol.strip().upper()}.NS"


def _csv_paths() -> list[Path]:
    directory = Path(settings.bulk_deals_dir)
    paths = sorted(directory.glob("*.csv")) if directory.exists() else []
    return paths


def _parse_file(path: Path) -> pd.DataFrame:
    # NSE exports carry a UTF-8 BOM and pad every header with a trailing space.
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise BulkDealsUnavailableError(f"{path.name} is missing columns: {sorted(missing)}.")

    frame = frame.rename(columns=COLUMN_RENAMES)
    frame["deal_date"] = pd.to_datetime(
        frame["deal_date"].str.strip(), format="%d-%b-%Y", errors="coerce"
    )
    frame["symbol"] = frame["symbol"].str.strip().str.upper()
    frame["security_name"] = frame["security_name"].str.strip()
    frame["client_name"] = frame["client_name"].str.strip()
    frame["side"] = frame["side"].str.strip().str.upper()
    # Quantities use Indian digit grouping, e.g. 3,90,000.
    frame["quantity"] = pd.to_numeric(
        frame["quantity"].str.replace(",", "", regex=False), errors="coerce"
    )
    frame["price"] = pd.to_numeric(
        frame["price"].str.replace(",", "", regex=False), errors="coerce"
    )

    usable = (
        frame["deal_date"].notna()
        & frame["symbol"].ne("")
        & frame["side"].isin(["BUY", "SELL"])
        & frame["quantity"].gt(0)
        & frame["price"].gt(0)
    )
    dropped = int((~usable).sum())
    if dropped:
        logger.warning("Dropped %s unusable bulk-deal row(s) from %s", dropped, path.name)
    frame = frame[usable].copy()
    frame["turnover"] = frame["quantity"] * frame["price"]
    frame["source_file"] = path.name
    return frame


_cache: tuple[tuple[tuple[str, float], ...], BulkDealsDataset] | None = None


def load_bulk_deals(force_reload: bool = False) -> BulkDealsDataset:
    """Parse every CSV in the bulk-deals directory, memoised by file mtime."""
    global _cache

    paths = _csv_paths()
    if not paths:
        raise BulkDealsUnavailableError(
            f"No bulk-deal CSV files found in {settings.bulk_deals_dir}. "
            "Export the NSE bulk deals report into that directory."
        )

    fingerprint = tuple((path.name, path.stat().st_mtime) for path in paths)
    if not force_reload and _cache is not None and _cache[0] == fingerprint:
        return _cache[1]

    frames = [_parse_file(path) for path in paths]
    deals = pd.concat(frames, ignore_index=True)
    # The same disclosure can appear twice if exported ranges overlap.
    deals = deals.drop_duplicates(
        subset=["deal_date", "symbol", "client_name", "side", "quantity", "price"]
    )
    if deals.empty:
        raise BulkDealsUnavailableError("The bulk-deal files contained no usable rows.")

    dataset = BulkDealsDataset(
        deals=deals.sort_values(["deal_date", "symbol"]).reset_index(drop=True),
        sources=tuple(path.name for path in paths),
    )
    _cache = (fingerprint, dataset)
    logger.info(
        "Loaded %s bulk-deal rows for %s symbols (%s to %s)",
        len(dataset.deals),
        len(dataset.symbols),
        dataset.start,
        dataset.end,
    )
    return dataset


def clear_cache() -> None:
    global _cache
    _cache = None
