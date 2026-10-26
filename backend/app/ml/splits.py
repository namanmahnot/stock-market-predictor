"""Chronological splitting and walk-forward origins (SRS section 9.1).

Time-series rows are never shuffled: the validation window always sits after
the training window, and the test window after the validation window.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """Half-open row ranges for the three chronological segments."""

    train_end: int
    validation_end: int
    n_rows: int

    @property
    def train(self) -> range:
        return range(0, self.train_end)

    @property
    def validation(self) -> range:
        return range(self.train_end, self.validation_end)

    @property
    def test(self) -> range:
        return range(self.validation_end, self.n_rows)

    def describe(self) -> dict[str, int]:
        return {
            "n_rows": self.n_rows,
            "n_train": len(self.train),
            "n_validation": len(self.validation),
            "n_test": len(self.test),
        }


def chronological_split(
    n_rows: int, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> Split:
    if n_rows < 10:
        raise ValueError(f"At least 10 rows are needed to split chronologically, got {n_rows}.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Fractions must lie strictly between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must leave rows for the test set.")

    train_end = int(round(n_rows * train_fraction))
    validation_end = int(round(n_rows * (train_fraction + validation_fraction)))
    # Guarantee at least one row in each later segment.
    train_end = min(max(train_end, 1), n_rows - 2)
    validation_end = min(max(validation_end, train_end + 1), n_rows - 1)
    return Split(train_end=train_end, validation_end=validation_end, n_rows=n_rows)


def walk_forward_origins(segment: range, step: int = 1, max_origins: int | None = None) -> list[int]:
    """Forecast origins inside an evaluation segment.

    ``step`` thins the origins so that a long history does not make model
    comparison unaffordable; the most recent origins are always kept because
    they describe the most relevant market regime.
    """
    if step < 1:
        raise ValueError("step must be at least 1.")
    origins = list(segment)[::step]
    if max_origins is not None and len(origins) > max_origins:
        origins = origins[-max_origins:]
    return origins
