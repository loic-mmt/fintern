from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fintern.metrics._base import (
    FundamentalsInput,
    MetricCandidate,
    MetricScaffoldBase,
)

_REVENUE_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "Revenues"),
    ("income_statement", "SalesRevenueNet"),
    ("income_statement", "RevenueFromContractWithCustomerExcludingAssessedTax"),
)
_NET_INCOME_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "NetIncomeLoss"),
    ("income_statement", "ProfitLoss"),
)
_EPS_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "EarningsPerShareDiluted"),
    ("income_statement", "EarningsPerShareBasic"),
)
_OPERATING_CASH_FLOW_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "NetCashProvidedByUsedInOperatingActivities"),
)
_CAPEX_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("cash_flow", "PaymentsToAcquireProductiveAssets"),
)
_BOOK_VALUE_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "StockholdersEquity"),
    (
        "balance_sheet",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
)


@dataclass(frozen=True)
class Growth(MetricScaffoldBase):
    """Calculate price- and fundamentals-driven growth metrics."""

    ticker: str
    prices: pd.Series | None = None
    data: pd.DataFrame | None = None
    fundamentals: FundamentalsInput = None

    @staticmethod
    def _validate_periods(periods: int) -> None:
        if not isinstance(periods, int):
            raise TypeError("periods must be an integer")

        if periods <= 0:
            raise ValueError("periods must be strictly positive")

    @classmethod
    def _growth_rate(
        cls,
        values: pd.Series,
        periods: int,
        name: str,
    ) -> pd.Series:
        cls._validate_periods(periods)

        if len(values) <= periods:
            raise ValueError("periods must be smaller than the number of observations")

        denominator = values.shift(periods)
        compared_denominators = denominator.dropna()

        if (compared_denominators == 0).any():
            raise ValueError(f"{name} is undefined when a prior value is zero")

        growth = values.div(denominator).sub(1).dropna()
        growth.name = name
        return growth

    def _fundamental_growth(
        self,
        candidates: tuple[MetricCandidate, ...],
        periods: int,
        name: str,
    ) -> pd.Series:
        values = self._fundamental_metric_series_from_candidates(
            candidates,
            date_column="period_end",
        )
        return self._growth_rate(values, periods=periods, name=name)

    def price_momentum(self, periods: int = 252) -> float:
        """Return trailing price momentum over a fixed number of periods."""
        self._validate_periods(periods)
        prices = self._close_prices()

        if len(prices) <= periods:
            raise ValueError("periods must be smaller than the number of prices")

        return float(prices.iloc[-1] / prices.iloc[-periods - 1] - 1)

    def rolling_price_momentum(self, periods: int = 252) -> pd.Series:
        """Return rolling trailing price momentum."""
        prices = self._close_prices()
        momentum = self._growth_rate(
            prices,
            periods=periods,
            name="price_momentum",
        )
        return momentum

    def annualized_price_growth(self, periods_per_year: float = 252.0) -> float:
        """Return annualized geometric price growth."""
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be strictly positive")

        prices = self._close_prices()

        if len(prices) < 2:
            raise ValueError("At least two prices are required")

        elapsed_periods = len(prices) - 1
        exponent = float(periods_per_year) / elapsed_periods
        return float((prices.iloc[-1] / prices.iloc[0]) ** exponent - 1)

    def revenue_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period revenue growth from fundamentals."""
        return self._fundamental_growth(
            _REVENUE_CANDIDATES,
            periods=periods,
            name="revenue_growth",
        )

    def net_income_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period net income growth from fundamentals."""
        return self._fundamental_growth(
            _NET_INCOME_CANDIDATES,
            periods=periods,
            name="net_income_growth",
        )

    def earnings_per_share_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period EPS growth from fundamentals."""
        return self._fundamental_growth(
            _EPS_CANDIDATES,
            periods=periods,
            name="earnings_per_share_growth",
        )

    def free_cash_flow_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period free cash flow growth."""
        operating_cash_flow = self._fundamental_metric_series_from_candidates(
            _OPERATING_CASH_FLOW_CANDIDATES,
            date_column="period_end",
        )
        capex = self._fundamental_metric_series_from_candidates(
            _CAPEX_CANDIDATES,
            date_column="period_end",
        )
        aligned = pd.concat(
            {
                "operating_cash_flow": operating_cash_flow,
                "capex": capex,
            },
            axis=1,
            join="inner",
        ).dropna()

        if aligned.empty:
            raise ValueError("No aligned operating cash flow and capex periods found")

        free_cash_flow = aligned["operating_cash_flow"] - aligned["capex"].abs()
        return self._growth_rate(
            free_cash_flow,
            periods=periods,
            name="free_cash_flow_growth",
        )

    def book_value_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period shareholders' equity growth."""
        return self._fundamental_growth(
            _BOOK_VALUE_CANDIDATES,
            periods=periods,
            name="book_value_growth",
        )


__all__ = ["Growth"]
