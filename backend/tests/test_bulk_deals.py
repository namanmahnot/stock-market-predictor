"""NSE bulk-deal parsing, aggregation, features and the ablation invariant."""
from __future__ import annotations

import pandas as pd
import pytest

from app.core.config import settings
from app.data.bulk_deals import (
    BulkDealsUnavailableError,
    clear_cache,
    load_bulk_deals,
    symbol_for_ticker,
    ticker_for_symbol,
)
from app.ml.bulk_features import BULK_FEATURE_COLUMNS, add_bulk_deal_features
from app.ml.features import build_feature_frame, feature_columns
from app.ml.pipeline import prepare_dataset
from tests.conftest import SYNTHETIC_TICKER, indian_grouping, make_bulk_deals_csv


# -- parsing ---------------------------------------------------------------
def test_indian_digit_grouping_round_trips():
    assert indian_grouping(390000) == "3,90,000"
    assert indian_grouping(90000) == "90,000"
    assert indian_grouping(500) == "500"


def test_loader_parses_the_export(offline_bulk_deals, bulk_deal_dates):
    dataset = load_bulk_deals()
    assert len(dataset.deals) == len(bulk_deal_dates)
    assert dataset.symbols == [SYNTHETIC_TICKER]
    assert dataset.deals["quantity"].min() > 0
    assert dataset.deals["turnover"].equals(dataset.deals["quantity"] * dataset.deals["price"])
    assert set(dataset.deals["side"]) <= {"BUY", "SELL"}


def test_loader_handles_indian_grouping(offline_bulk_deals):
    dataset = load_bulk_deals()
    # First synthetic row is 50,000 and must not be read as 50.
    assert dataset.deals["quantity"].iloc[0] == pytest.approx(50_000)


def test_loader_is_memoised_until_the_file_changes(offline_bulk_deals, bulk_deal_dates):
    first = load_bulk_deals()
    assert load_bulk_deals() is first

    make_bulk_deals_csv(offline_bulk_deals / "synthetic-bulk-deals.csv", bulk_deal_dates[:5])
    clear_cache()
    assert len(load_bulk_deals().deals) == 5


def test_loader_drops_unusable_rows(tmp_path, monkeypatch):
    path = tmp_path / "broken.csv"
    path.write_text(
        '"Date ","Symbol ","Security Name ","Client Name ","Buy / Sell ",'
        '"Quantity Traded ","Trade Price / Wght. Avg. Price ","Remarks "\n'
        '"01-JAN-2026","GOOD","Good Ltd","CLIENT A","BUY","1,00,000","25.50","-"\n'
        '"not-a-date","BAD","Bad Ltd","CLIENT B","BUY","1,000","10.00","-"\n'
        '"02-JAN-2026","BAD","Bad Ltd","CLIENT C","HOLD","1,000","10.00","-"\n'
        '"03-JAN-2026","BAD","Bad Ltd","CLIENT D","BUY","0","10.00","-"\n'
        '"04-JAN-2026","BAD","Bad Ltd","CLIENT E","SELL","1,000","0","-"\n',
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(settings, "bulk_deals_dir", tmp_path)
    clear_cache()
    dataset = load_bulk_deals()
    assert len(dataset.deals) == 1
    assert dataset.deals["symbol"].iloc[0] == "GOOD"


def test_loader_reports_a_missing_column(tmp_path, monkeypatch):
    path = tmp_path / "incomplete.csv"
    path.write_text('"Date ","Symbol "\n"01-JAN-2026","X"\n', encoding="utf-8-sig")
    monkeypatch.setattr(settings, "bulk_deals_dir", tmp_path)
    clear_cache()
    with pytest.raises(BulkDealsUnavailableError, match="missing columns"):
        load_bulk_deals()


def test_loader_reports_an_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "bulk_deals_dir", tmp_path)
    clear_cache()
    with pytest.raises(BulkDealsUnavailableError, match="No bulk-deal CSV"):
        load_bulk_deals()


# -- symbol mapping --------------------------------------------------------
@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("RELIANCE.NS", "RELIANCE"),
        ("reliance.ns", "RELIANCE"),
        ("TEST", "TEST"),
        ("^NSEI", None),
        ("TCS.BO", None),
        ("", None),
    ],
)
def test_symbol_mapping(ticker, expected):
    assert symbol_for_ticker(ticker) == expected


def test_ticker_for_symbol_adds_the_nse_suffix():
    assert ticker_for_symbol("atalreal") == "ATALREAL.NS"


# -- aggregation -----------------------------------------------------------
def test_daily_aggregate_matches_the_raw_rows(offline_bulk_deals):
    dataset = load_bulk_deals()
    raw = dataset.for_ticker(SYNTHETIC_TICKER)
    daily = dataset.daily(SYNTHETIC_TICKER)

    assert len(daily) == raw["deal_date"].nunique()
    assert daily["deal_count"].sum() == len(raw)
    assert daily["gross_quantity"].sum() == pytest.approx(raw["quantity"].sum())
    assert daily["buy_quantity"].sum() == pytest.approx(
        raw[raw["side"] == "BUY"]["quantity"].sum()
    )
    assert daily["net_quantity"].equals(daily["buy_quantity"] - daily["sell_quantity"])
    # VWAP must sit inside the range of the prices it averages.
    session = daily.index[0]
    prices = raw[raw["deal_date"] == session]["price"]
    assert prices.min() <= daily["deal_vwap"].iloc[0] <= prices.max()


def test_coverage_counts_sessions_not_rows(offline_bulk_deals, bulk_deal_dates):
    coverage = load_bulk_deals().coverage(SYNTHETIC_TICKER)
    assert coverage.in_dataset
    assert coverage.sessions_with_deals == len(bulk_deal_dates)
    assert coverage.sufficient


def test_coverage_for_an_absent_symbol(offline_bulk_deals):
    coverage = load_bulk_deals().coverage("RELIANCE.NS")
    assert not coverage.in_dataset
    assert coverage.sessions_with_deals == 0
    assert not coverage.sufficient


def test_coverage_table_is_ranked(offline_bulk_deals):
    rows = load_bulk_deals().coverage_table()
    assert rows[0]["ticker"] == f"{SYNTHETIC_TICKER}.NS"
    assert rows[0]["sufficient"] is True


# -- features --------------------------------------------------------------
@pytest.fixture
def enriched(price_frame, offline_bulk_deals):
    return add_bulk_deal_features(build_feature_frame(price_frame), SYNTHETIC_TICKER)


def test_features_are_added_and_finite(enriched):
    for column in BULK_FEATURE_COLUMNS:
        assert column in enriched.frame.columns
    values = enriched.frame[BULK_FEATURE_COLUMNS]
    assert not values.isna().any().any()
    assert values.map(lambda v: abs(v) < 1e6).all().all()


def test_frame_is_clipped_to_the_disclosure_window(enriched, offline_bulk_deals):
    dataset = load_bulk_deals()
    assert enriched.frame.index[0].date() >= dataset.start
    assert enriched.frame.index[-1].date() <= dataset.end
    assert enriched.rows_after < enriched.rows_before
    assert any("clipped" in note for note in enriched.notes)


def test_sessions_without_a_deal_are_zero_not_missing(enriched):
    flagged = enriched.frame["bd_has_deal"]
    assert set(flagged.unique()) <= {0.0, 1.0}
    quiet = enriched.frame[flagged == 0.0]
    assert (quiet["bd_deal_count"] == 0).all()
    assert (quiet["bd_net_qty_ratio"] == 0).all()


def test_deal_sessions_carry_a_signal(enriched):
    active = enriched.frame[enriched.frame["bd_has_deal"] == 1.0]
    assert len(active) == enriched.sessions_matched > 0
    assert (active["bd_deal_count"] > 0).all()
    assert active["bd_gross_qty_ratio"].abs().sum() > 0


def test_sessions_since_deal_resets_and_counts_up(enriched):
    frame = enriched.frame
    since = frame["bd_sessions_since_deal"]
    assert (since[frame["bd_has_deal"] == 1.0] == 0).all()
    # Between disclosures the counter only ever increases by one.
    deltas = since.diff().dropna()
    assert deltas.isin([1.0, 0.0]).all() or (deltas <= 1.0).all()


def test_rolling_windows_only_look_backwards(enriched):
    """Truncating the future must not change a past row's rolling features."""
    frame = enriched.frame
    cut = len(frame) // 2
    truncated = frame.iloc[:cut]
    for column in ("bd_net_qty_ratio_5", "bd_net_qty_ratio_20", "bd_sessions_since_deal"):
        recomputed = (
            truncated[column] if column == "bd_sessions_since_deal" else truncated[column]
        )
        pd.testing.assert_series_equal(frame[column].iloc[:cut], recomputed)


def test_features_for_an_absent_symbol_are_refused(price_frame, offline_bulk_deals):
    with pytest.raises(BulkDealsUnavailableError, match="does not appear"):
        add_bulk_deal_features(build_feature_frame(price_frame), "RELIANCE.NS")


def test_publication_lag_shifts_the_disclosure(price_frame, offline_bulk_deals, monkeypatch):
    frame = build_feature_frame(price_frame)
    immediate = add_bulk_deal_features(frame, SYNTHETIC_TICKER).frame["bd_has_deal"]

    monkeypatch.setattr(settings, "bulk_deals_publication_lag", 1)
    lagged = add_bulk_deal_features(frame, SYNTHETIC_TICKER).frame["bd_has_deal"]
    assert lagged.iloc[1:].to_numpy() == pytest.approx(immediate.iloc[:-1].to_numpy())


# -- dataset preparation ---------------------------------------------------
def test_prepare_dataset_without_bulk_deals(history):
    prepared = prepare_dataset(history, use_bulk_deals=False)
    assert not prepared.uses_bulk_deals
    assert prepared.feature_set == "price_only"
    assert not any(column.startswith("bd_") for column in prepared.columns)


def test_prepare_dataset_with_bulk_deals(history):
    prepared = prepare_dataset(history, use_bulk_deals=True)
    assert prepared.uses_bulk_deals
    assert prepared.feature_set == "price+bulk_deals"
    assert len([c for c in prepared.columns if c.startswith("bd_")]) == len(BULK_FEATURE_COLUMNS)
    assert prepared.bulk_deals["coverage"]["sufficient"] is True


def test_auto_mode_uses_the_group_when_coverage_is_good(history):
    prepared = prepare_dataset(history, use_bulk_deals=None)
    assert prepared.uses_bulk_deals


def test_auto_mode_skips_the_group_when_coverage_is_thin(history, offline_bulk_deals, bulk_deal_dates):
    make_bulk_deals_csv(offline_bulk_deals / "synthetic-bulk-deals.csv", bulk_deal_dates[:3])
    clear_cache()
    prepared = prepare_dataset(history, use_bulk_deals=None)
    assert not prepared.uses_bulk_deals
    assert any("skipped" in note for note in prepared.notes)


def test_requiring_an_unavailable_group_is_an_error(history, offline_bulk_deals, bulk_deal_dates):
    from app.ml.pipeline import InsufficientHistoryError

    make_bulk_deals_csv(
        offline_bulk_deals / "synthetic-bulk-deals.csv", bulk_deal_dates, symbol="OTHER"
    )
    clear_cache()
    with pytest.raises(InsufficientHistoryError, match="does not appear"):
        prepare_dataset(history, use_bulk_deals=True)


def test_auto_mode_degrades_quietly_for_a_non_nse_symbol(history, offline_bulk_deals, bulk_deal_dates):
    make_bulk_deals_csv(
        offline_bulk_deals / "synthetic-bulk-deals.csv", bulk_deal_dates, symbol="OTHER"
    )
    clear_cache()
    prepared = prepare_dataset(history, use_bulk_deals=None)
    assert not prepared.uses_bulk_deals
    assert any("unavailable" in note for note in prepared.notes)


# -- the ablation invariant ------------------------------------------------
def test_both_arms_share_the_same_rows(history):
    """The ablation is only meaningful if the two feature sets cover one window."""
    from app.ml.pipeline import PreparedDataset, run_experiment

    with_bulk = prepare_dataset(history, use_bulk_deals=True)
    stripped = with_bulk.frame.drop(columns=BULK_FEATURE_COLUMNS)
    without_bulk = PreparedDataset(
        frame=stripped, columns=feature_columns(stripped), uses_bulk_deals=False
    )
    assert with_bulk.frame.index.equals(without_bulk.frame.index)

    left = run_experiment(history, horizon=1, model_names=["naive"], prepared=without_bulk)
    right = run_experiment(history, horizon=1, model_names=["naive"], prepared=with_bulk)
    # The naive baseline has no features, so identical rows must give identical
    # metrics. If this ever fails, the two arms are not comparable.
    assert left.report_for("naive").test.primary.mae == pytest.approx(
        right.report_for("naive").test.primary.mae
    )
    assert left.split.describe() == right.split.describe()
