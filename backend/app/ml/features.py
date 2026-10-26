"""Technical indicators and the supervised learning matrix (SRS section 7).

Leakage rule enforced here: every feature on row ``t`` is computed only from
observations up to and including session ``t``, and every target describes
sessions strictly after ``t``. A forecast produced after the close of session
``t`` therefore never consumes a price it could not have known.

The learning target is the cumulative log return over the next ``h`` sessions
rather than the raw price. Predicting the price level directly lets any model
score well by echoing the last close, which hides whether it has learned
anything; returns make the comparison against the naive baseline meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Longest look-back used by an indicator; rows before this are warm-up only.
WARMUP_ROWS = 60

FEATURE_COLUMNS: list[str] = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "mom_10",
    "mom_20",
    "sma_5_ratio",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "rsi_14",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "volatility_10",
    "volatility_20",
    "range_pct",
    "close_position",
    "volume_change",
    "volume_ratio_20",
    "gap_pct",
]

INDICATOR_COLUMNS: list[str] = [
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "daily_return",
    "volatility_20",
    "volume_change",
]


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def relative_strength_index(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI; 100 when there are no losses in the window."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rsi = pd.Series(np.nan, index=series.index, dtype="float64")
    nonzero = avg_loss > 0
    rs = avg_gain[nonzero] / avg_loss[nonzero]
    rsi[nonzero] = 100.0 - (100.0 / (1.0 + rs))
    flat = (avg_loss == 0) & avg_gain.notna()
    rsi[flat] = np.where(avg_gain[flat] > 0, 100.0, 50.0)
    return rsi


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = exponential_moving_average(series, fast) - exponential_moving_average(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def rolling_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    return log_returns.rolling(window=window, min_periods=window).std()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Append technical indicators and model features to an OHLCV frame."""
    out = frame.copy()
    close = out["adj_close"].astype("float64")
    log_close = np.log(close)
    log_return = log_close.diff()

    out["daily_return"] = close.pct_change()
    out["log_return"] = log_return

    for window in (5, 10, 20, 50):
        out[f"sma_{window}"] = simple_moving_average(close, window)
        out[f"sma_{window}_ratio"] = close / out[f"sma_{window}"] - 1.0
    for span in (12, 26):
        out[f"ema_{span}"] = exponential_moving_average(close, span)
        out[f"ema_{span}_ratio"] = close / out[f"ema_{span}"] - 1.0

    # Kept in natural 0-100 units for display; models that need comparable
    # scales standardise their inputs during fitting.
    out["rsi_14"] = relative_strength_index(close, 14)
    macd_line, signal_line, hist = macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    out["macd_norm"] = macd_line / close
    out["macd_signal_norm"] = signal_line / close
    out["macd_hist_norm"] = hist / close

    # ret_1 is the most recent completed session return; ret_k is k-1 sessions older.
    for lag in (1, 2, 3, 5, 10):
        out[f"ret_{lag}"] = log_return.shift(lag - 1)
    out["mom_10"] = log_close - log_close.shift(10)
    out["mom_20"] = log_close - log_close.shift(20)

    out["volatility_10"] = rolling_volatility(log_return, 10)
    out["volatility_20"] = rolling_volatility(log_return, 20)

    high, low, open_ = out["high"].astype("float64"), out["low"].astype("float64"), out["open"].astype("float64")
    raw_close = out["close"].astype("float64")
    out["range_pct"] = (high - low) / raw_close
    span = (high - low).replace(0.0, np.nan)
    out["close_position"] = ((raw_close - low) / span).fillna(0.5)
    out["gap_pct"] = open_ / raw_close.shift(1) - 1.0

    volume = out["volume"].astype("float64")
    out["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    volume_mean = volume.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
    out["volume_ratio_20"] = volume / volume_mean

    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model input columns present in ``frame``, in a fixed order.

    Optional feature groups (currently the NSE bulk-deal group) only appear
    when they were attached, so a model trained with them records a longer
    feature list and one trained without them is unaffected.
    """
    from app.ml.bulk_features import BULK_FEATURE_COLUMNS

    return [column for column in FEATURE_COLUMNS + BULK_FEATURE_COLUMNS if column in frame.columns]


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Indicators plus warm-up removal, ready for model fitting."""
    enriched = add_indicators(frame)
    enriched = enriched.replace([np.inf, -np.inf], np.nan)
    cleaned = enriched.dropna(subset=FEATURE_COLUMNS)
    return cleaned


def build_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Cumulative log return from session ``t`` to ``t+h`` for h = 1..horizon."""
    log_close = np.log(frame["adj_close"].astype("float64"))
    targets = {f"y_{h}": log_close.shift(-h) - log_close for h in range(1, horizon + 1)}
    return pd.DataFrame(targets, index=frame.index)


def build_supervised(
    frame: pd.DataFrame, horizon: int, drop_incomplete: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, targets) aligned on the same index.

    With ``drop_incomplete`` the trailing rows whose future is not yet known are
    removed, which is what training needs. Inference keeps them and reads the
    final row.
    """
    features = frame[feature_columns(frame)]
    targets = build_targets(frame, horizon)
    if drop_incomplete:
        mask = targets.notna().all(axis=1)
        return features[mask], targets[mask]
    return features, targets


def prices_from_log_returns(reference_price: float, cumulative_log_returns: np.ndarray) -> np.ndarray:
    """Convert cumulative log returns back to a price path."""
    return float(reference_price) * np.exp(np.asarray(cumulative_log_returns, dtype="float64"))


def next_trading_days(last_date: pd.Timestamp, steps: int) -> list[pd.Timestamp]:
    """Future business days, used to label forecast steps as trading sessions.

    Weekends are excluded; exchange holidays are not modelled, so a target date
    is an estimate of the n-th future session rather than a calendar guarantee.
    """
    dates: list[pd.Timestamp] = []
    cursor = pd.Timestamp(last_date)
    while len(dates) < steps:
        cursor = cursor + pd.Timedelta(days=1)
        if cursor.weekday() < 5:
            dates.append(cursor)
    return dates
