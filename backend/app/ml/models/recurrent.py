"""LSTM and GRU sequence models (SRS 6.1, items 5 and 6).

Both models read a sliding window of recent sessions and emit the cumulative
log return for every horizon step in one pass. Scaling statistics and the
early-stopping split are derived from training rows only.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.ml.base import BaseForecaster, NotEnoughDataError
from app.ml.features import FEATURE_COLUMNS, build_supervised

logger = logging.getLogger(__name__)


def _build_network(cell: str, n_features: int, hidden_size: int, num_layers: int, horizon: int, dropout: float):
    import torch.nn as nn

    class RecurrentForecastNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, horizon))

        def forward(self, x):
            output, _ = self.rnn(x)
            return self.head(output[:, -1, :])

    return RecurrentForecastNet()


class RecurrentForecaster(BaseForecaster):
    """Shared training loop for the LSTM and GRU variants."""

    cell: str = "lstm"
    refit_cost = "high"
    min_train_rows = 150

    def __init__(
        self,
        horizon: int = 1,
        window: int = 30,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 120,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 15,
        **params: Any,
    ) -> None:
        super().__init__(
            horizon=horizon,
            window=window,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            **params,
        )

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "PyTorch is not installed (pip install torch)."
        return True, None

    # -- windowing ---------------------------------------------------------
    def _windows(self, features: np.ndarray, targets: np.ndarray | None, window: int):
        n = features.shape[0]
        if n < window:
            raise NotEnoughDataError(
                f"{self.display_name} needs at least {window} sessions to form one window."
            )
        xs = np.stack([features[i - window : i] for i in range(window, n + 1)])
        if targets is None:
            return xs, None
        ys = targets[window - 1 :]
        return xs, ys

    def _fit(self, frame: pd.DataFrame) -> None:
        import torch

        torch.manual_seed(settings.random_seed)
        np.random.seed(settings.random_seed)

        window = int(self.params["window"])
        features, targets = build_supervised(frame, self.horizon, drop_incomplete=True)
        if len(features) < window + 40:
            raise NotEnoughDataError(
                f"{self.display_name} needs {window + 40} rows with a complete future, "
                f"only {len(features)} are available."
            )
        self.feature_names_ = list(features.columns)
        x_raw = features.to_numpy(dtype="float64")
        y_raw = targets.to_numpy(dtype="float64")

        self.x_mean_ = x_raw.mean(axis=0)
        self.x_std_ = np.where(x_raw.std(axis=0) < 1e-12, 1.0, x_raw.std(axis=0))
        # Targets are tiny log returns; scaling them keeps the loss well conditioned.
        self.y_std_ = float(max(y_raw.std(), 1e-6))

        x_scaled = (x_raw - self.x_mean_) / self.x_std_
        xs, ys = self._windows(x_scaled, y_raw / self.y_std_, window)

        # Chronological early-stopping tail inside the training segment.
        holdout = max(int(len(xs) * 0.15), 10)
        if len(xs) - holdout < 30:
            holdout = max(len(xs) // 5, 1)
        x_tr = torch.tensor(xs[:-holdout], dtype=torch.float32)
        y_tr = torch.tensor(ys[:-holdout], dtype=torch.float32)
        x_va = torch.tensor(xs[-holdout:], dtype=torch.float32)
        y_va = torch.tensor(ys[-holdout:], dtype=torch.float32)

        net = _build_network(
            self.cell,
            n_features=x_tr.shape[2],
            hidden_size=int(self.params["hidden_size"]),
            num_layers=int(self.params["num_layers"]),
            horizon=self.horizon,
            dropout=float(self.params["dropout"]),
        )
        optimiser = torch.optim.Adam(net.parameters(), lr=float(self.params["learning_rate"]))
        loss_fn = torch.nn.MSELoss()
        batch_size = int(self.params["batch_size"])
        patience = int(self.params["patience"])

        best_loss, best_state, waited = float("inf"), None, 0
        for epoch in range(int(self.params["epochs"])):
            net.train()
            order = torch.randperm(x_tr.shape[0])
            for start in range(0, x_tr.shape[0], batch_size):
                idx = order[start : start + batch_size]
                optimiser.zero_grad()
                loss = loss_fn(net(x_tr[idx]), y_tr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimiser.step()

            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(x_va), y_va))
            if val_loss < best_loss - 1e-7:
                best_loss, waited = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                if waited >= patience:
                    logger.debug("%s early stopped at epoch %s", self.display_name, epoch)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.params["best_internal_val_loss"] = best_loss
        self.window_ = window

    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        import torch

        window = self.window_
        rows = frame[self.feature_names_].to_numpy(dtype="float64")[-window:]
        if rows.shape[0] < window:
            raise NotEnoughDataError(
                f"{self.display_name} needs {window} recent sessions to predict."
            )
        scaled = (rows - self.x_mean_) / self.x_std_
        tensor = torch.tensor(scaled[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            output = self.net_(tensor).numpy().reshape(-1)
        return output * self.y_std_

    def feature_names(self) -> list[str]:
        return list(getattr(self, "feature_names_", FEATURE_COLUMNS))


class LSTMForecaster(RecurrentForecaster):
    name = "lstm"
    display_name = "LSTM"
    family = "deep_learning"
    cell = "lstm"
    description = (
        "Recurrent network over a 30-session window of the engineered features. "
        "Learns sequential structure directly instead of relying on hand-picked lags."
    )


class GRUForecaster(RecurrentForecaster):
    name = "gru"
    display_name = "GRU"
    family = "deep_learning"
    cell = "gru"
    description = (
        "Gated recurrent network with a simpler gating structure than the LSTM. "
        "Trains faster and is included as an optional deep-learning comparison."
    )
