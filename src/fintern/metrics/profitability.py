from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fintern.metrics._base import FundamentalsInput, MetricScaffoldBase


@dataclass(frozen=True)
class Profitability(MetricScaffoldBase):
    """Skeleton for fundamentals-driven profitability metrics.

    These metrics mainly depend on normalized statement rows and, in a few
    cases, on mixing income statement items with balance-sheet items.

    Candidate first implementations:
    - gross margin
    - operating margin
    - net margin
    - return on assets
    - return on equity
    - return on invested capital
    - free cash flow margin
    """

    ticker: str
    prices: pd.Series | None = None
    data: pd.DataFrame | None = None
    fundamentals: FundamentalsInput = None

    # TODO(fintern): gross_profit / revenue using aligned fiscal periods.
    def gross_margin(self) -> pd.Series:
        """Return gross margin by reported period."""
        raise NotImplementedError("TODO: implement gross margin")

    # TODO(fintern): operating_income / revenue by aligned fiscal period.
    def operating_margin(self) -> pd.Series:
        """Return operating margin by reported period."""
        raise NotImplementedError("TODO: implement operating margin")

    # TODO(fintern): net_income / revenue by aligned fiscal period.
    def net_margin(self) -> pd.Series:
        """Return net margin by reported period."""
        raise NotImplementedError("TODO: implement net margin")

    # TODO(fintern): ebit / revenue once EBIT mapping is stabilized.
    def ebit_margin(self) -> pd.Series:
        """Return EBIT margin by reported period."""
        raise NotImplementedError("TODO: implement EBIT margin")

    # TODO(fintern): operating_cash_flow - capex, then divide by revenue.
    def free_cash_flow_margin(self) -> pd.Series:
        """Return free cash flow margin by reported period."""
        raise NotImplementedError("TODO: implement free cash flow margin")

    # TODO(fintern): net_income divided by average total assets.
    def return_on_assets(self) -> pd.Series:
        """Return return on assets by reported period."""
        raise NotImplementedError("TODO: implement return on assets")

    # TODO(fintern): net_income divided by average shareholders' equity.
    def return_on_equity(self) -> pd.Series:
        """Return return on equity by reported period."""
        raise NotImplementedError("TODO: implement return on equity")

    # TODO(fintern): NOPAT divided by average invested capital.
    def return_on_invested_capital(self) -> pd.Series:
        """Return ROIC by reported period."""
        raise NotImplementedError("TODO: implement return on invested capital")


__all__ = ["Profitability"]
