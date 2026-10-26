"""ARIMA / SARIMAX statistical benchmark (SRS 6.1)."""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from app.ml.base import BaseForecaster, NotEnoughDataError

logger = logging.getLogger(__name__)

#: Small, fixed order grid. Orders are selected by AIC on the training rows
#: only, so the choice never sees validation or test data.
ORDER_GRID: tuple[tuple[int, int, int], ...] = (
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
)


class ArimaForecaster(BaseForecaster):
    """ARIMA fitted on the log of the adjusted close.

    Modelling log prices with d = 1 means the model works on log returns, the
    same quantity the other models predict, and keeps the forecast positive.
    The order is chosen by AIC over a small grid rather than a wide automatic
    search, which keeps a walk-forward evaluation affordable.
    """

    name = "arima"
    display_name = "ARIMA"
    family = "statistical"
    description = (
        "Classical time-series benchmark on log prices. Captures autocorrelation and "
        "drift without any technical indicators, which makes it a clean reference "
        "for whether the engineered features add anything."
    )
    refit_cost = "high"
    min_train_rows = 80

    def __init__(self, horizon: int = 1, order: tuple[int, int, int] | None = None, **params: Any) -> None:
        super().__init__(horizon=horizon, order=order, **params)
        self.order_: tuple[int, int, int] | None = tuple(order) if order else None

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            return False, "statsmodels is not installed (pip install statsmodels)."
        return True, None

    def _fit(self, frame: pd.DataFrame) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        series = np.log(frame["adj_close"].astype("float64").to_numpy())
        if not np.isfinite(series).all():
            raise NotEnoughDataError("ARIMA requires a finite log-price series.")

        candidates = [self.order_] if self.order_ else list(ORDER_GRID)
        best_aic, best_order, best_fit = np.inf, None, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in candidates:
                try:
                    # With d = 1 statsmodels rejects a constant, so a linear
                    # trend is used: on the differenced log price that is the
                    # drift term, which is what a trending series needs.
                    fitted = ARIMA(series, order=order, trend="t").fit()
                except Exception:
                    logger.debug("ARIMA order %s failed to converge", order, exc_info=True)
                    continue
                if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                    best_aic, best_order, best_fit = float(fitted.aic), order, fitted

        if best_fit is None:
            raise NotEnoughDataError("No ARIMA order in the grid converged on this series.")
        self.order_ = best_order
        self.params["order"] = list(best_order)
        self.params["aic"] = best_aic
        self.result_ = best_fit
        self.train_last_log_ = float(series[-1])
        self.train_series_ = series

    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        series = np.log(frame["adj_close"].astype("float64").to_numpy())
        result = self.result_
        # Rolling origins past the training window are handled by appending the
        # newly observed sessions without re-estimating the coefficients.
        if series.size > self.train_series_.size:
            extra = series[self.train_series_.size :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = result.append(extra, refit=False)
                except Exception:
                    logger.debug("ARIMA append failed; forecasting from the fitted window")
        forecast = np.asarray(result.forecast(steps=self.horizon), dtype="float64")
        return forecast - float(series[-1])

    def hyperparameters(self) -> dict[str, Any]:
        params = dict(self.params)
        params["order"] = list(self.order_) if self.order_ else None
        return params
