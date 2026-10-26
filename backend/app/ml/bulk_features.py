"""Features derived from NSE bulk-deal disclosures (SRS section 7.6).

Every column here describes session *t* using only deals disclosed for session
*t* or earlier, so the leakage rule that governs the price features governs
these too. Rolling windows are backward-looking sums.

The feature frame is clipped to the window the dataset actually covers. Padding
the earlier years with zeros would be the tempting shortcut and would be wrong:
before the file starts, "no deal" means "not recorded", and a model told
otherwise would learn that bulk deals began the day the export did.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.core.config import settings
from app.data.bulk_deals import (
    BulkDealsDataset,
    BulkDealsUnavailableError,
    Coverage,
    load_bulk_deals,
)

logger = logging.getLogger(__name__)

BULK_FEATURE_COLUMNS: list[str] = [
    "bd_has_deal",
    "bd_deal_count",
    "bd_net_qty_ratio",
    "bd_gross_qty_ratio",
    "bd_turnover_ratio",
    "bd_imbalance",
    "bd_price_premium",
    "bd_net_qty_ratio_5",
    "bd_net_qty_ratio_20",
    "bd_sessions_since_deal",
]

#: Cap for "sessions since the last disclosed deal", so the feature stays bounded.
MAX_SESSIONS_SINCE = 60


@dataclass
class BulkFeatureResult:
    frame: pd.DataFrame
    coverage: Coverage
    sessions_matched: int
    rows_before: int
    notes: list[str] = field(default_factory=list)

    @property
    def rows_after(self) -> int:
        return len(self.frame)

    def as_dict(self) -> dict:
        return {
            "coverage": self.coverage.as_dict(),
            "sessions_matched": self.sessions_matched,
            "rows_before_clipping": self.rows_before,
            "rows_after_clipping": self.rows_after,
            "features": BULK_FEATURE_COLUMNS,
            "publication_lag_sessions": settings.bulk_deals_publication_lag,
            "notes": self.notes,
        }


def add_bulk_deal_features(
    frame: pd.DataFrame,
    ticker: str,
    dataset: BulkDealsDataset | None = None,
) -> BulkFeatureResult:
    """Append bulk-deal features to an engineered price frame.

    ``frame`` is the indicator frame, which still carries ``close`` and
    ``volume``; the returned frame is clipped to the dataset's covered window.
    """
    dataset = dataset or load_bulk_deals()
    coverage = dataset.coverage(ticker)
    notes: list[str] = []
    rows_before = len(frame)

    if not coverage.in_dataset:
        raise BulkDealsUnavailableError(
            f"{ticker} does not appear in the bulk-deal dataset "
            f"({coverage.dataset_start} to {coverage.dataset_end}). "
            "Large-cap symbols rarely do, because a single deal would have to move more "
            "than 0.5% of the company's equity."
        )

    out = frame.copy()
    daily = dataset.daily(ticker)

    # Clip to the window the dataset describes: outside it, "no deal" is unknown.
    window_start = pd.Timestamp(coverage.dataset_start)
    window_end = pd.Timestamp(coverage.dataset_end)
    in_window = (out.index >= window_start) & (out.index <= window_end)
    out = out[in_window]
    if out.empty:
        raise BulkDealsUnavailableError(
            f"The price history for {ticker} does not overlap the bulk-deal window "
            f"({coverage.dataset_start} to {coverage.dataset_end})."
        )
    notes.append(
        f"Dataset clipped to the bulk-deal coverage window {out.index[0].date()} to "
        f"{out.index[-1].date()} ({len(out)} of {rows_before} sessions)."
    )

    # Align disclosures to trading sessions present in the price history.
    aligned = daily.reindex(out.index).fillna(0.0)
    unmatched = int(len(daily) - int((daily.index.isin(out.index)).sum()))
    if unmatched > 0:
        notes.append(
            f"{unmatched} disclosed session(s) had no matching price bar in this window and were ignored."
        )

    lag = int(settings.bulk_deals_publication_lag)
    if lag > 0:
        aligned = aligned.shift(lag).fillna(0.0)
        notes.append(f"Disclosures treated as known {lag} session(s) after the deal date.")

    volume = out["volume"].astype("float64").replace(0.0, np.nan)
    close = out["close"].astype("float64")
    session_turnover = (volume * close).replace(0.0, np.nan)

    out["bd_has_deal"] = (aligned["deal_count"] > 0).astype("float64")
    out["bd_deal_count"] = aligned["deal_count"].astype("float64")
    out["bd_net_qty_ratio"] = (aligned["net_quantity"] / volume).fillna(0.0)
    out["bd_gross_qty_ratio"] = (aligned["gross_quantity"] / volume).fillna(0.0)
    out["bd_turnover_ratio"] = (aligned["gross_turnover"] / session_turnover).fillna(0.0)

    gross_quantity = aligned["gross_quantity"].replace(0.0, np.nan)
    out["bd_imbalance"] = (aligned["net_quantity"] / gross_quantity).fillna(0.0)

    premium = (aligned["deal_vwap"] / close - 1.0).where(aligned["deal_count"] > 0)
    out["bd_price_premium"] = premium.fillna(0.0)

    # Backward-looking sums: the pressure from recent disclosures, which decays.
    out["bd_net_qty_ratio_5"] = out["bd_net_qty_ratio"].rolling(5, min_periods=1).sum()
    out["bd_net_qty_ratio_20"] = out["bd_net_qty_ratio"].rolling(20, min_periods=1).sum()
    out["bd_sessions_since_deal"] = _sessions_since(out["bd_has_deal"])

    out[BULK_FEATURE_COLUMNS] = (
        out[BULK_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    matched = int((aligned["deal_count"] > 0).sum())
    notes.append(f"{matched} session(s) in this window carry a disclosed bulk deal.")
    if matched < settings.bulk_deals_min_sessions:
        notes.append(
            f"Only {matched} session(s) with deals: below the configured minimum of "
            f"{settings.bulk_deals_min_sessions}, so these features carry very little signal."
        )

    return BulkFeatureResult(
        frame=out,
        coverage=coverage,
        sessions_matched=matched,
        rows_before=rows_before,
        notes=notes,
    )


def _sessions_since(has_deal: pd.Series) -> pd.Series:
    """Sessions elapsed since the last disclosed deal, counted causally."""
    counts = np.empty(len(has_deal), dtype="float64")
    since = float(MAX_SESSIONS_SINCE)
    for i, flagged in enumerate(has_deal.to_numpy()):
        if flagged > 0:
            since = 0.0
        elif since < MAX_SESSIONS_SINCE:
            since += 1.0
        counts[i] = since
    return pd.Series(counts, index=has_deal.index)
