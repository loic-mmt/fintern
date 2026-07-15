from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fintern.data.periods import FiscalFrequency
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
_GROSS_PROFIT_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "GrossProfit"),
)
_COST_OF_REVENUE_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "CostOfGoodsSold"),
    ("income_statement", "CostOfRevenue"),
)
_OPERATING_INCOME_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "OperatingIncomeLoss"),
)
_NET_INCOME_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "NetIncomeLoss"),
    ("income_statement", "ProfitLoss"),
)
_OPERATING_CASH_FLOW_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "NetCashProvidedByUsedInOperatingActivities"),
)
_CAPEX_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("cash_flow", "PaymentsToAcquireProductiveAssets"),
)
_ASSETS_CANDIDATES: tuple[MetricCandidate, ...] = (("balance_sheet", "Assets"),)
_EQUITY_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "StockholdersEquity"),
    (
        "balance_sheet",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
)
_TOTAL_DEBT_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "LongTermDebtAndCapitalLeaseObligations"),
    ("balance_sheet", "DebtAndFinanceLeaseObligations"),
    ("balance_sheet", "LongTermDebt"),
)
_CURRENT_DEBT_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
    ("balance_sheet", "DebtAndFinanceLeaseObligationsCurrent"),
    ("balance_sheet", "LongTermDebtCurrentMaturities"),
    ("balance_sheet", "ShortTermBorrowings"),
    ("balance_sheet", "CommercialPaper"),
)
_NONCURRENT_DEBT_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "LongTermDebtAndCapitalLeaseObligationsNoncurrent"),
    ("balance_sheet", "DebtAndFinanceLeaseObligationsNoncurrent"),
    ("balance_sheet", "LongTermDebtNoncurrent"),
)
_CASH_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "CashAndCashEquivalentsAtCarryingValue"),
)
_TAX_EXPENSE_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "IncomeTaxExpenseBenefit"),
)
_PRETAX_INCOME_CANDIDATES: tuple[MetricCandidate, ...] = (
    (
        "income_statement",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ),
    (
        "income_statement",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
)


@dataclass(frozen=True)
class Profitability(MetricScaffoldBase):
    """Calculate fundamentals-driven profitability metrics."""

    ticker: str
    prices: pd.Series | None = None
    data: pd.DataFrame | None = None
    fundamentals: FundamentalsInput = None
    as_of: str | pd.Timestamp | None = None
    frequency: FiscalFrequency = "quarterly"

    def _series(self, candidates: tuple[MetricCandidate, ...]) -> pd.Series:
        return self._fundamental_metric_series_from_candidates(
            candidates,
            date_column="period_end",
            frequency=self.frequency,
        )

    @staticmethod
    def _align_series(**series: pd.Series) -> pd.DataFrame:
        aligned = pd.concat(series, axis=1, join="inner").dropna()

        if aligned.empty:
            names = ", ".join(series)
            raise ValueError(f"No aligned fundamentals periods found for: {names}")

        return aligned

    @classmethod
    def _ratio_series(
        cls,
        numerator: pd.Series,
        denominator: pd.Series,
        name: str,
    ) -> pd.Series:
        aligned = cls._align_series(
            numerator=numerator,
            denominator=denominator,
        )

        if (aligned["denominator"] == 0).any():
            raise ValueError(f"{name} is undefined when denominator is zero")

        result = aligned["numerator"].div(aligned["denominator"])
        result.name = name
        return result

    def _revenue(self) -> pd.Series:
        return self._series(_REVENUE_CANDIDATES)

    def _gross_profit(self) -> pd.Series:
        try:
            return self._series(_GROSS_PROFIT_CANDIDATES)
        except ValueError:
            aligned = self._align_series(
                revenue=self._revenue(),
                cost_of_revenue=self._series(_COST_OF_REVENUE_CANDIDATES),
            )
            gross_profit = aligned["revenue"] - aligned["cost_of_revenue"].abs()
            gross_profit.name = "gross_profit"
            return gross_profit

    def _free_cash_flow(self) -> pd.Series:
        aligned = self._align_series(
            operating_cash_flow=self._series(_OPERATING_CASH_FLOW_CANDIDATES),
            capex=self._series(_CAPEX_CANDIDATES),
        )
        free_cash_flow = aligned["operating_cash_flow"] - aligned["capex"].abs()
        free_cash_flow.name = "free_cash_flow"
        return free_cash_flow

    def _total_debt(self) -> pd.Series:
        try:
            return self._series(_TOTAL_DEBT_CANDIDATES)
        except ValueError:
            aligned = self._align_series(
                current_debt=self._series(_CURRENT_DEBT_CANDIDATES),
                noncurrent_debt=self._series(_NONCURRENT_DEBT_CANDIDATES),
            )
            total_debt = aligned["current_debt"] + aligned["noncurrent_debt"]
            total_debt.name = "total_debt"
            return total_debt

    @classmethod
    def _return_on_average_balance(
        cls,
        income: pd.Series,
        balance: pd.Series,
        name: str,
    ) -> pd.Series:
        average_balance = balance.add(balance.shift(1)).div(2).dropna()

        if average_balance.empty:
            raise ValueError(f"{name} requires at least two balance observations")

        return cls._ratio_series(income, average_balance, name=name)

    def gross_margin(self) -> pd.Series:
        """Return gross margin by reported period."""
        return self._ratio_series(
            self._gross_profit(),
            self._revenue(),
            name="gross_margin",
        )

    def operating_margin(self) -> pd.Series:
        """Return operating margin by reported period."""
        return self._ratio_series(
            self._series(_OPERATING_INCOME_CANDIDATES),
            self._revenue(),
            name="operating_margin",
        )

    def net_margin(self) -> pd.Series:
        """Return net margin by reported period."""
        return self._ratio_series(
            self._series(_NET_INCOME_CANDIDATES),
            self._revenue(),
            name="net_margin",
        )

    def ebit_margin(self) -> pd.Series:
        """Return EBIT margin using operating income as EBIT."""
        return self._ratio_series(
            self._series(_OPERATING_INCOME_CANDIDATES),
            self._revenue(),
            name="ebit_margin",
        )

    def free_cash_flow_margin(self) -> pd.Series:
        """Return free cash flow margin by reported period."""
        return self._ratio_series(
            self._free_cash_flow(),
            self._revenue(),
            name="free_cash_flow_margin",
        )

    def return_on_assets(self) -> pd.Series:
        """Return net income divided by average total assets."""
        return self._return_on_average_balance(
            self._series(_NET_INCOME_CANDIDATES),
            self._series(_ASSETS_CANDIDATES),
            name="return_on_assets",
        )

    def return_on_equity(self) -> pd.Series:
        """Return net income divided by average shareholders' equity."""
        return self._return_on_average_balance(
            self._series(_NET_INCOME_CANDIDATES),
            self._series(_EQUITY_CANDIDATES),
            name="return_on_equity",
        )

    def _effective_tax_rate(self) -> pd.Series:
        tax_expense = self._series(_TAX_EXPENSE_CANDIDATES)
        pretax_income = self._series(_PRETAX_INCOME_CANDIDATES)
        tax_rate = self._ratio_series(
            tax_expense,
            pretax_income,
            name="effective_tax_rate",
        )

        if ((tax_rate < 0) | (tax_rate >= 1)).any():
            raise ValueError(
                "effective tax rate must be between 0 and 1; pass tax_rate "
                "explicitly for non-standard periods"
            )

        return tax_rate

    def return_on_invested_capital(
        self,
        tax_rate: float | None = None,
    ) -> pd.Series:
        """Return NOPAT divided by average invested capital.

        When ``tax_rate`` is omitted, effective tax rate is derived from tax
        expense and pretax income for each reported period.
        """
        if tax_rate is not None and not 0 <= tax_rate < 1:
            raise ValueError("tax_rate must be greater than or equal to 0 and below 1")

        operating_income = self._series(_OPERATING_INCOME_CANDIDATES)
        rates = (
            pd.Series(float(tax_rate), index=operating_income.index)
            if tax_rate is not None
            else self._effective_tax_rate()
        )
        nopat_inputs = self._align_series(
            operating_income=operating_income,
            tax_rate=rates,
        )
        nopat = nopat_inputs["operating_income"] * (1 - nopat_inputs["tax_rate"])

        invested_inputs = self._align_series(
            equity=self._series(_EQUITY_CANDIDATES),
            debt=self._total_debt(),
            cash=self._series(_CASH_CANDIDATES),
        )
        invested_capital = (
            invested_inputs["equity"]
            + invested_inputs["debt"]
            - invested_inputs["cash"]
        )
        average_invested_capital = (
            invested_capital.add(invested_capital.shift(1)).div(2).dropna()
        )

        if average_invested_capital.empty:
            raise ValueError("ROIC requires at least two invested-capital observations")

        return self._ratio_series(
            nopat,
            average_invested_capital,
            name="return_on_invested_capital",
        )


__all__ = ["Profitability"]
