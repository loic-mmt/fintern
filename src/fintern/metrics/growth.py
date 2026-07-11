from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fintern.metrics._base import FundamentalsInput, MetricScaffoldBase


@dataclass(frozen=True)
class Growth(MetricScaffoldBase):
    """Skeleton for price- and fundamentals-driven growth metrics.

    Price metrics that can be implemented with the current market data layer:
    - trailing price momentum
    - rolling price momentum
    - annualized price growth

    Fundamentals metrics that can be implemented with normalized statements:
    - revenue growth
    - net income growth
    - EPS growth
    - free cash flow growth
    - book value growth
    """

    ticker: str
    prices: pd.Series | None = None
    data: pd.DataFrame | None = None
    fundamentals: FundamentalsInput = None

    # TODO(fintern): implement from self._close_prices() using trailing window.
    def price_momentum(self, periods: int = 252) -> float:
        """Return trailing price momentum over a fixed number of periods."""
        raise NotImplementedError("TODO: implement trailing price momentum")

    # TODO(fintern): implement as a Series based on self._close_prices().
    def rolling_price_momentum(self, periods: int = 252) -> pd.Series:
        """Return rolling trailing price momentum."""
        raise NotImplementedError("TODO: implement rolling price momentum")

    # TODO(fintern): implement from self._close_prices() with periods_per_year.
    def annualized_price_growth(self, periods_per_year: float = 252.0) -> float:
        """Return annualized geometric price growth."""
        raise NotImplementedError("TODO: implement annualized price growth")

    # TODO(fintern): map revenue metric and compute pct_change(periods).
    def revenue_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period revenue growth from fundamentals."""
        raise NotImplementedError("TODO: implement revenue growth")

    # TODO(fintern): map net income metric and compute pct_change(periods).
    def net_income_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period net income growth from fundamentals."""
        raise NotImplementedError("TODO: implement net income growth")

    # TODO(fintern): map EPS metric and compute pct_change(periods).
    def earnings_per_share_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period EPS growth from fundamentals."""
        raise NotImplementedError("TODO: implement EPS growth")

    # TODO(fintern): map cash-flow metrics and derive free cash flow first.
    def free_cash_flow_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period free cash flow growth."""
        raise NotImplementedError("TODO: implement free cash flow growth")

    # TODO(fintern): map equity or BVPS metric and compute pct_change(periods).
    def book_value_growth(self, periods: int = 1) -> pd.Series:
        """Return period-over-period book value growth."""
        raise NotImplementedError("TODO: implement book value growth")


__all__ = ["Growth"]
