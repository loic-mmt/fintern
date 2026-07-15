"""Historical analysis of market prices and reported fundamentals.

Use ``period_end`` to study how a fundamental metric evolved between reporting
periods. Use ``filed_date`` when combining fundamentals with market prices so a
value cannot be used before it became public.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from fintern.data.periods import FiscalFrequency
from fintern.metrics._base import FundamentalsInput, MetricScaffoldBase

HistoryDate = Literal["period_end", "filed_date"]


@dataclass(frozen=True)
class HistoricalAnalysis(MetricScaffoldBase):
    """Build historical views from normalized prices and fundamentals.

    Price/fundamental extraction, growth analysis, normalized trends,
    point-in-time alignment, snapshots, and compact summaries share the same
    validated input model as the metrics package.
    """

    ticker: str
    prices: pd.Series | None = None
    data: pd.DataFrame | None = None
    fundamentals: FundamentalsInput = None
    as_of: str | pd.Timestamp | None = None
    frequency: FiscalFrequency | None = None

    @staticmethod
    def _date_bounds(
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Normalize and validate optional inclusive date bounds."""
        start_date = pd.Timestamp(start) if start is not None else None
        end_date = pd.Timestamp(end) if end is not None else None

        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("start must be before or equal to end")

        return start_date, end_date

    @staticmethod
    def _slice_dates(
        values: pd.Series,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> pd.Series:
        """Return an inclusive date slice without relying on string slicing."""
        result = values

        if start is not None:
            result = result.loc[result.index >= start]

        if end is not None:
            result = result.loc[result.index <= end]

        if result.empty:
            raise ValueError("No observations found inside the requested date range")

        return result.copy()

    def price_history(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.Series:
        """Return sorted close prices inside an inclusive date range."""
        start_date, end_date = self._date_bounds(start, end)
        prices = self._close_prices()

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("price history requires a DatetimeIndex")

        return self._slice_dates(prices.sort_index(), start_date, end_date)

    def fundamental_history(
        self,
        metric: str,
        statement: str | None = None,
        *,
        date_by: HistoryDate = "period_end",
        frequency: FiscalFrequency | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.Series:
        """Return one normalized fundamental metric inside a date range.

        ``period_end`` represents the economic reporting period. ``filed_date``
        represents when the information became public.
        """
        if date_by not in {"period_end", "filed_date"}:
            raise ValueError("date_by must be either `period_end` or `filed_date`")

        start_date, end_date = self._date_bounds(start, end)
        values = self._fundamental_metric_series(
            metric=metric,
            statement=statement,
            date_column=date_by,
            frequency=frequency or self.frequency,
        )
        return self._slice_dates(values, start_date, end_date)

    def period_change(
        self,
        metric: str,
        periods: int = 1,
        statement: str | None = None,
        *,
        frequency: FiscalFrequency | None = None,
    ) -> pd.Series:
        """Return fractional period-over-period changes for one metric."""
        if isinstance(periods, bool) or not isinstance(periods, int):
            raise TypeError("periods must be an integer")

        if periods < 1:
            raise ValueError("periods must be strictly positive")

        values = self.fundamental_history(
            metric=metric,
            statement=statement,
            date_by="period_end",
            frequency=frequency,
        )

        if len(values) <= periods:
            raise ValueError("periods must be smaller than the number of observations")

        prior_values = values.shift(periods)

        if (prior_values.dropna() == 0).any():
            raise ValueError(f"{metric} change is undefined when a prior value is zero")

        changes = values.pct_change(periods=periods, fill_method=None).dropna()
        changes.name = f"{metric}_change"
        return changes

    @staticmethod
    def _calculate_cagr(values: pd.Series, metric: str) -> float:
        """Calculate CAGR from a dated series using actual elapsed time."""
        if len(values) < 2:
            raise ValueError(f"{metric} CAGR requires at least two observations")

        first_value = float(values.iloc[0])
        last_value = float(values.iloc[-1])

        if first_value <= 0 or last_value <= 0:
            raise ValueError(f"{metric} CAGR requires strictly positive endpoints")

        elapsed_seconds = (values.index[-1] - values.index[0]).total_seconds()
        elapsed_years = elapsed_seconds / (365.2425 * 24 * 60 * 60)

        if elapsed_years <= 0:
            raise ValueError(f"{metric} CAGR requires distinct chronological dates")

        return float((last_value / first_value) ** (1 / elapsed_years) - 1)

    def compound_annual_growth_rate(
        self,
        metric: str,
        statement: str | None = None,
        *,
        frequency: FiscalFrequency | None = None,
    ) -> float:
        """Return CAGR between the first and last reported values.

        Elapsed years come from actual period-end dates. CAGR is undefined for
        zero or negative endpoints, so those inputs raise ``ValueError``.
        """
        values = self.fundamental_history(
            metric=metric,
            statement=statement,
            date_by="period_end",
            frequency=frequency,
        )
        return self._calculate_cagr(values, metric)

    def trend(
        self,
        metric: str,
        statement: str | None = None,
        *,
        frequency: FiscalFrequency | None = None,
    ) -> float:
        """Return the metric's normalized annual linear trend.

        The linear-regression slope is measured per elapsed year and divided by
        the mean absolute metric value. Positive results indicate an upward
        trend, negative results a downward trend, and magnitude is comparable
        across metrics with different units.
        """
        values = self.fundamental_history(
            metric=metric,
            statement=statement,
            date_by="period_end",
            frequency=frequency,
        )

        if len(values) < 3:
            raise ValueError(f"{metric} trend requires at least three observations")

        observations = values.to_numpy(dtype=float)

        if not np.isfinite(observations).all():
            raise ValueError(f"{metric} trend requires finite values")

        elapsed_years = np.asarray(
            (values.index - values.index[0]).total_seconds(),
            dtype=float,
        ) / (365.2425 * 24 * 60 * 60)
        centered_time = elapsed_years - elapsed_years.mean()
        centered_values = observations - observations.mean()
        time_variance = float(np.dot(centered_time, centered_time))

        if time_variance == 0:
            raise ValueError(f"{metric} trend requires distinct dates")

        scale = float(np.mean(np.abs(observations)))

        if scale == 0:
            raise ValueError(f"{metric} trend is undefined when its scale is zero")

        slope = float(np.dot(centered_time, centered_values) / time_variance)
        return slope / scale

    def stability(
        self,
        metric: str,
        statement: str | None = None,
        *,
        frequency: FiscalFrequency | None = None,
    ) -> float:
        """Return a growth-stability score between zero and one.

        The score is ``1 / (1 + sample standard deviation of period changes)``.
        One means perfectly stable growth; increasingly volatile growth moves
        the score toward zero. Negative changes and sign changes are supported.
        """
        changes = self.period_change(
            metric=metric,
            periods=1,
            statement=statement,
            frequency=frequency,
        )

        if len(changes) < 2:
            raise ValueError(f"{metric} stability requires at least three observations")

        dispersion = float(changes.std(ddof=1))

        if not np.isfinite(dispersion):
            raise ValueError(f"{metric} stability requires finite period changes")

        return float(1 / (1 + dispersion))

    def _point_in_time_metric_events(
        self,
        metric: str,
        statement: str | None,
    ) -> pd.DataFrame:
        """Return latest-period metric values as they became public."""
        rows = self._ticker_statements()
        required_columns = {"filed_date", "period_end"}
        missing_columns = sorted(required_columns - set(rows.columns))

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"fundamentals statements must contain columns: {missing}")

        rows = rows.loc[rows["metric"].astype(str) == metric].copy()

        if statement is not None:
            if "statement" not in rows.columns:
                raise ValueError(
                    "fundamentals statements must contain a `statement` column"
                )
            rows = rows.loc[rows["statement"].astype(str) == statement].copy()

        if rows.empty:
            raise ValueError(
                f"No fundamentals rows found for metric={metric!r} "
                f"and ticker={self.ticker}"
            )

        rows["filed_date"] = pd.to_datetime(rows["filed_date"], errors="coerce")
        rows["period_end"] = pd.to_datetime(rows["period_end"], errors="coerce")
        rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
        rows = rows.dropna(subset=["filed_date", "period_end", "value"])

        if self.as_of is not None:
            rows = rows.loc[rows["filed_date"] <= pd.Timestamp(self.as_of)].copy()

        if rows.empty:
            raise ValueError(
                f"No dated values found for metric={metric!r} and ticker={self.ticker}"
            )

        rows = rows.sort_values(["filed_date", "period_end"])
        latest_period = rows["period_end"].cummax()
        events = rows.loc[rows["period_end"] == latest_period].copy()
        events = events.drop_duplicates(subset=["filed_date"], keep="last")
        return events.sort_values("filed_date")

    def align_fundamental_with_prices(
        self,
        metric: str,
        statement: str | None = None,
    ) -> pd.DataFrame:
        """Align each price with the latest fundamental known on that date.

        The backward as-of join uses filing dates. Prices before the first filing
        remain in the result with a missing fundamental value. Later amendments
        update the value only when they concern the latest reported period.
        """
        prices = self.price_history().rename("close")
        market = prices.rename_axis("date").reset_index().sort_values("date")
        events = self._point_in_time_metric_events(metric, statement)
        published = (
            events.loc[:, ["filed_date", "value"]]
            .rename(columns={"filed_date": "date"})
            .sort_values("date")
        )
        aligned = pd.merge_asof(
            market,
            published,
            on="date",
            direction="backward",
        ).set_index("date")
        aligned.index.name = None
        return aligned.loc[:, ["close", "value"]]

    def snapshot(self, as_of: str | pd.Timestamp) -> pd.DataFrame:
        """Return the latest version of every metric known by ``as_of``.

        For each statement/metric pair, the latest available reporting period is
        selected, followed by the latest filing for that period. Filing metadata
        is retained so the result remains auditable.
        """
        as_of_date = pd.Timestamp(as_of)

        if pd.isna(as_of_date):
            raise ValueError("as_of must be a valid date")

        rows = self._ticker_statements()
        required_columns = {"statement", "metric", "filed_date", "period_end"}
        missing_columns = sorted(required_columns - set(rows.columns))

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"fundamentals statements must contain columns: {missing}")

        rows["filed_date"] = pd.to_datetime(rows["filed_date"], errors="coerce")
        rows["period_end"] = pd.to_datetime(rows["period_end"], errors="coerce")
        rows = rows.dropna(subset=["filed_date", "period_end"])
        rows = rows.loc[rows["filed_date"] <= as_of_date].copy()

        if rows.empty:
            raise ValueError(
                f"No fundamentals were available for ticker={self.ticker} "
                f"as of {as_of_date.date()}"
            )

        group_columns = ["statement", "metric"]
        snapshot = (
            rows.sort_values(group_columns + ["period_end", "filed_date"])
            .drop_duplicates(subset=group_columns, keep="last")
            .sort_values(group_columns)
            .reset_index(drop=True)
        )
        return snapshot

    def summary(
        self,
        metrics: list[str],
        *,
        frequency: FiscalFrequency | None = None,
    ) -> pd.DataFrame:
        """Return latest, prior, change, CAGR, and count for each metric.

        Missing metrics raise ``ValueError``. A metric with only one observation
        is retained, with missing prior/change/CAGR values.
        """
        if not isinstance(metrics, list):
            raise TypeError("metrics must be a list of metric names")

        if not metrics:
            raise ValueError("metrics cannot be empty")

        if any(not isinstance(metric, str) or not metric.strip() for metric in metrics):
            raise ValueError("metrics must contain non-empty strings")

        summary_rows = []

        for metric in metrics:
            values = self.fundamental_history(
                metric,
                date_by="period_end",
                frequency=frequency,
            )
            observations = len(values)
            latest = float(values.iloc[-1])
            prior = float(values.iloc[-2]) if observations >= 2 else np.nan
            change = latest / prior - 1 if observations >= 2 and prior != 0 else np.nan
            cagr = (
                self._calculate_cagr(values, metric)
                if observations >= 2 and float(values.iloc[0]) > 0 and latest > 0
                else np.nan
            )
            summary_rows.append(
                {
                    "metric": metric,
                    "latest": latest,
                    "prior": prior,
                    "change": change,
                    "cagr": cagr,
                    "observations": observations,
                }
            )

        return pd.DataFrame(summary_rows)


__all__ = ["HistoricalAnalysis", "HistoryDate"]
