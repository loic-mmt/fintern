from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fintern.metrics._base import (
    FundamentalsInput,
    MetricCandidate,
    MetricScaffoldBase,
)

_SHARES_OUTSTANDING_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "CommonStockSharesOutstanding"),
)
_REVENUE_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "Revenues"),
    ("income_statement", "SalesRevenueNet"),
    ("income_statement", "RevenueFromContractWithCustomerExcludingAssessedTax"),
)
_NET_INCOME_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "NetIncomeLoss"),
)
_OPERATING_INCOME_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "OperatingIncomeLoss"),
)
_EPS_DILUTED_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "EarningsPerShareDiluted"),
)
_EPS_BASIC_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "EarningsPerShareBasic"),
)
_EQUITY_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "StockholdersEquity"),
)
_CASH_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "CashAndCashEquivalentsAtCarryingValue"),
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
_OPERATING_CASH_FLOW_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "NetCashProvidedByUsedInOperatingActivities"),
)
_CAPEX_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "PaymentsToAcquirePropertyPlantAndEquipment"),
)
_EBITDA_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "EarningsBeforeInterestTaxesDepreciationAndAmortization"),
)
_DEPRECIATION_AND_AMORTIZATION_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "DepreciationDepletionAndAmortization"),
    ("cash_flow", "DepreciationAmortizationAndAccretionNet"),
)
_DEPRECIATION_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "Depreciation"),
)
_AMORTIZATION_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "AmortizationOfIntangibleAssets"),
)


@dataclass(frozen=True)
class Valuation(MetricScaffoldBase):
    """Skeleton for valuation metrics mixing prices and fundamentals.

    Combined price + fundamentals metrics that fit the current data model:
    - market capitalization
    - price-to-earnings
    - price-to-sales
    - price-to-book
    - enterprise value
    - EV / EBITDA
    - EV / sales
    - free cash flow yield
    """

    ticker: str
    prices: pd.Series | None = None
    data: pd.DataFrame | None = None
    fundamentals: FundamentalsInput = None

    def _latest_shares_outstanding(self) -> float:
        """Return latest available shares outstanding."""
        return self._latest_fundamental_value_from_candidates(
            _SHARES_OUTSTANDING_CANDIDATES
        )

    def _latest_revenue(self) -> float:
        """Return latest available revenue using ordered metric fallbacks."""
        return self._latest_fundamental_value_from_candidates(_REVENUE_CANDIDATES)

    def _latest_net_income(self) -> float:
        """Return latest available net income."""
        return self._latest_fundamental_value_from_candidates(_NET_INCOME_CANDIDATES)

    def _latest_operating_income(self) -> float:
        """Return latest available operating income."""
        return self._latest_fundamental_value_from_candidates(
            _OPERATING_INCOME_CANDIDATES
        )

    def _latest_eps(self) -> float:
        """Return diluted EPS when available, else fall back to basic EPS."""
        return self._latest_fundamental_value_from_candidates(
            _EPS_DILUTED_CANDIDATES + _EPS_BASIC_CANDIDATES
        )

    def _latest_equity(self) -> float:
        """Return latest available shareholders' equity."""
        return self._latest_fundamental_value_from_candidates(_EQUITY_CANDIDATES)

    def _latest_cash(self) -> float:
        """Return latest available cash and cash equivalents."""
        return self._latest_fundamental_value_from_candidates(_CASH_CANDIDATES)

    def _latest_operating_cash_flow(self) -> float:
        """Return latest available operating cash flow."""
        return self._latest_fundamental_value_from_candidates(
            _OPERATING_CASH_FLOW_CANDIDATES
        )

    def _latest_capex(self) -> float:
        """Return latest available capital expenditures fact."""
        return self._latest_fundamental_value_from_candidates(_CAPEX_CANDIDATES)

    def _latest_depreciation_and_amortization(self) -> float:
        """Return latest available depreciation and amortization charge."""
        try:
            return self._latest_fundamental_value_from_candidates(
                _DEPRECIATION_AND_AMORTIZATION_CANDIDATES
            )
        except ValueError:
            depreciation = self._latest_fundamental_value_from_candidates(
                _DEPRECIATION_CANDIDATES
            )
            amortization = self._latest_fundamental_value_from_candidates(
                _AMORTIZATION_CANDIDATES
            )
            return float(depreciation + amortization)

    @staticmethod
    def _divide_or_raise(
        numerator: float,
        denominator: float,
        metric_name: str,
    ) -> float:
        """Return ratio while rejecting zero denominators."""
        if denominator == 0:
            raise ValueError(f"{metric_name} is undefined when denominator is zero")

        return float(numerator / denominator)

    def _latest_total_debt(self) -> float:
        """Return latest available debt using ordered direct and fallback metrics."""
        try:
            return self._latest_fundamental_value_from_candidates(
                _TOTAL_DEBT_CANDIDATES
            )
        except ValueError:
            current_debt = self._latest_fundamental_value_from_candidates(
                _CURRENT_DEBT_CANDIDATES
            )
            noncurrent_debt = self._latest_fundamental_value_from_candidates(
                _NONCURRENT_DEBT_CANDIDATES
            )
            return float(current_debt + noncurrent_debt)

    def _latest_ebitda(self) -> float:
        """Return latest available EBITDA, direct or derived from EBIT + D&A."""
        try:
            return self._latest_fundamental_value_from_candidates(_EBITDA_CANDIDATES)
        except ValueError:
            operating_income = self._latest_operating_income()
            depreciation_and_amortization = (
                self._latest_depreciation_and_amortization()
            )
            return float(operating_income + depreciation_and_amortization)

    def market_cap(self) -> float:
        """Return market capitalization from price and share count."""
        price = self._latest_close_price()
        shares = self._latest_shares_outstanding()

        if shares <= 0:
            raise ValueError("shares outstanding must be strictly positive")

        return float(price * shares)

    def price_to_sales(self) -> float:
        """Return price-to-sales using the latest available revenue."""
        market_cap = self.market_cap()
        revenue = self._latest_revenue()
        return self._divide_or_raise(
            market_cap,
            revenue,
            "price_to_sales",
        )

    def price_to_earnings(self) -> float:
        """Return price-to-earnings using the latest available earnings."""
        try:
            net_income = self._latest_net_income()
            return self._divide_or_raise(
                self.market_cap(),
                net_income,
                "price_to_earnings",
            )
        except ValueError as net_income_error:
            try:
                eps = self._latest_eps()
            except ValueError:
                raise net_income_error from None

            return self._divide_or_raise(
                self._latest_close_price(),
                eps,
                "price_to_earnings",
            )

    def price_to_book(self) -> float:
        """Return price-to-book using the latest book value."""
        market_cap = self.market_cap()
        equity = self._latest_equity()
        return self._divide_or_raise(
            market_cap,
            equity,
            "price_to_book",
        )

    def enterprise_value(self) -> float:
        """Return enterprise value."""
        market_cap = self.market_cap()
        total_debt = self._latest_total_debt()
        cash = self._latest_cash()
        return float(market_cap + total_debt - cash)

    def ev_to_ebitda(self) -> float:
        """Return EV/EBITDA."""
        enterprise_value = self.enterprise_value()
        ebitda = self._latest_ebitda()
        return self._divide_or_raise(
            enterprise_value,
            ebitda,
            "ev_to_ebitda",
        )

    def ev_to_sales(self) -> float:
        """Return EV/sales."""
        enterprise_value = self.enterprise_value()
        revenue = self._latest_revenue()
        return self._divide_or_raise(
            enterprise_value,
            revenue,
            "ev_to_sales",
        )

    def free_cash_flow_yield(self) -> float:
        """Return free cash flow yield."""
        operating_cash_flow = self._latest_operating_cash_flow()
        capex = abs(self._latest_capex())
        free_cash_flow = operating_cash_flow - capex
        market_cap = self.market_cap()
        return self._divide_or_raise(
            free_cash_flow,
            market_cap,
            "free_cash_flow_yield",
        )


__all__ = ["Valuation"]
