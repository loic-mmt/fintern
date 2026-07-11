from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

FundamentalsInput = pd.DataFrame | Mapping[str, pd.DataFrame] | None
MetricCandidate = tuple[str | None, str]


@dataclass(frozen=True)
class MetricScaffoldBase:
    """Shared input validation and extraction helpers for metric scaffolds."""

    ticker: str
    prices: pd.Series | None = None
    data: pd.DataFrame | None = None
    fundamentals: FundamentalsInput = None

    def __post_init__(self) -> None:
        normalized_ticker = self.ticker.strip().upper()

        if not normalized_ticker:
            raise ValueError("ticker cannot be empty")

        object.__setattr__(self, "ticker", normalized_ticker)

        if self.prices is not None:
            if not isinstance(self.prices, pd.Series):
                raise TypeError("prices must be a pandas Series")

            if self.prices.empty:
                raise ValueError("prices cannot be empty")

            if self.prices.isna().any():
                raise ValueError("prices cannot contain missing values")

            if (self.prices <= 0).any():
                raise ValueError("prices must be strictly positive")

            object.__setattr__(self, "prices", self.prices.astype(float))

        if self.data is not None:
            if not isinstance(self.data, pd.DataFrame):
                raise TypeError("data must be a pandas DataFrame")

            if self.data.empty:
                raise ValueError("data cannot be empty")

        if self.fundamentals is not None:
            self._statements_frame()

    def _close_prices(self) -> pd.Series:
        """Return a normalized close-price series for the current ticker."""
        if self.prices is not None:
            close_prices = self.prices.copy()

            if isinstance(close_prices.index, pd.DatetimeIndex):
                close_prices = close_prices.sort_index()

            return close_prices.astype(float)

        if self.data is None:
            raise ValueError("prices or data with a close column are required")

        required_columns = {"date", "ticker", "close"}
        missing_columns = sorted(required_columns - set(self.data.columns))

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"data must contain columns: {missing}")

        ticker_data = self.data.loc[
            self.data["ticker"].astype(str).str.upper() == self.ticker,
            ["date", "close"],
        ].copy()

        if ticker_data.empty:
            raise ValueError(f"No close-price data found for ticker={self.ticker}")

        ticker_data["date"] = pd.to_datetime(ticker_data["date"])
        ticker_data = ticker_data.sort_values("date").set_index("date")
        close_prices = ticker_data["close"].astype(float)

        if close_prices.isna().any():
            raise ValueError("close prices cannot contain missing values")

        if (close_prices <= 0).any():
            raise ValueError("close prices must be strictly positive")

        return close_prices

    def _statements_frame(self) -> pd.DataFrame:
        """Return the normalized fundamentals statements table."""
        if self.fundamentals is None:
            raise ValueError("fundamentals data is required for this metric")

        if isinstance(self.fundamentals, pd.DataFrame):
            statements = self.fundamentals.copy()
        elif isinstance(self.fundamentals, Mapping):
            statements = self.fundamentals.get("statements", pd.DataFrame()).copy()
        else:
            raise TypeError(
                "fundamentals must be a DataFrame or a mapping of DataFrames"
            )

        if not isinstance(statements, pd.DataFrame):
            raise TypeError("fundamentals statements payload must be a DataFrame")

        if statements.empty:
            raise ValueError("fundamentals statements cannot be empty")

        required_columns = {"ticker", "metric", "value"}
        missing_columns = sorted(required_columns - set(statements.columns))

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(
                "fundamentals statements must contain columns: "
                f"{missing}"
            )

        return statements

    def _ticker_statements(self) -> pd.DataFrame:
        """Return fundamentals rows filtered to the current ticker."""
        statements = self._statements_frame()
        ticker_rows = statements.loc[
            statements["ticker"].astype(str).str.upper() == self.ticker
        ].copy()

        if ticker_rows.empty:
            raise ValueError(
                f"No fundamentals statement data found for ticker={self.ticker}"
            )

        return ticker_rows

    def _fundamental_metric_series(
        self,
        metric: str,
        statement: str | None = None,
    ) -> pd.Series:
        """Return a time series for one normalized fundamentals metric."""
        ticker_rows = self._ticker_statements()

        metric_rows = ticker_rows.loc[
            ticker_rows["metric"].astype(str) == metric
        ].copy()

        if statement is not None:
            if "statement" not in metric_rows.columns:
                raise ValueError(
                    "fundamentals statements must contain a `statement` column"
                )

            metric_rows = metric_rows.loc[
                metric_rows["statement"].astype(str) == statement
            ].copy()

        if metric_rows.empty:
            raise ValueError(
                f"No fundamentals rows found for metric={metric!r} "
                f"and ticker={self.ticker}"
            )

        date_column = "filed_date"
        if (
            date_column not in metric_rows.columns
            or metric_rows[date_column].isna().all()
        ):
            date_column = "period_end"

        if date_column not in metric_rows.columns:
            raise ValueError(
                "fundamentals statements must contain `filed_date` or `period_end`"
            )

        metric_rows[date_column] = pd.to_datetime(metric_rows[date_column])
        metric_rows = metric_rows.dropna(subset=[date_column, "value"])

        if metric_rows.empty:
            raise ValueError(
                f"No dated values found for metric={metric!r} "
                f"and ticker={self.ticker}"
            )

        metric_rows = metric_rows.sort_values(date_column)
        series = (
            metric_rows.drop_duplicates(subset=[date_column], keep="last")
            .set_index(date_column)["value"]
            .astype(float)
        )
        series.index = pd.DatetimeIndex(series.index.to_numpy())
        series.index.name = None
        series.name = metric

        return series

    def _latest_fundamental_value(
        self,
        metric: str,
        statement: str | None = None,
    ) -> float:
        """Return the latest available value for one fundamentals metric."""
        series = self._fundamental_metric_series(metric=metric, statement=statement)
        return float(series.iloc[-1])

    def _fundamental_metric_series_from_candidates(
        self,
        candidates: Sequence[MetricCandidate],
    ) -> pd.Series:
        """Return first available fundamentals series from ordered candidates."""
        if not candidates:
            raise ValueError("candidates cannot be empty")

        last_error: ValueError | None = None

        for statement, metric in candidates:
            try:
                return self._fundamental_metric_series(
                    metric=metric,
                    statement=statement,
                )
            except ValueError as exc:
                last_error = exc

        metrics = ", ".join(metric for _, metric in candidates)
        raise ValueError(
            "No fundamentals rows found for any candidate metrics: "
            f"{metrics}"
        ) from last_error

    def _latest_fundamental_value_from_candidates(
        self,
        candidates: Sequence[MetricCandidate],
    ) -> float:
        """Return latest value from first available candidate metric."""
        series = self._fundamental_metric_series_from_candidates(candidates)
        return float(series.iloc[-1])

    def _latest_close_price(self) -> float:
        """Return the latest available close price."""
        close_prices = self._close_prices()
        return float(close_prices.iloc[-1])


__all__ = [
    "FundamentalsInput",
    "MetricCandidate",
    "MetricScaffoldBase",
]
