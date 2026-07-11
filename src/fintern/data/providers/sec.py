from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import pandas as pd

from fintern.data.exceptions import InstrumentResolutionError
from fintern.data.models import NormalizedFundamentals
from fintern.data.providers.base import ProviderBase

_STATEMENT_BY_METRIC = {
    "Revenues": "income_statement",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "income_statement",
    "SalesRevenueNet": "income_statement",
    "CostOfGoodsSold": "income_statement",
    "GrossProfit": "income_statement",
    "OperatingIncomeLoss": "income_statement",
    "NetIncomeLoss": "income_statement",
    "EarningsPerShareBasic": "income_statement",
    "EarningsPerShareDiluted": "income_statement",
    "EarningsBeforeInterestTaxesDepreciationAndAmortization": "income_statement",
    "Assets": "balance_sheet",
    "AssetsCurrent": "balance_sheet",
    "Liabilities": "balance_sheet",
    "LiabilitiesCurrent": "balance_sheet",
    "StockholdersEquity": "balance_sheet",
    "CommonStockSharesOutstanding": "balance_sheet",
    "CashAndCashEquivalentsAtCarryingValue": "balance_sheet",
    "LongTermDebtAndCapitalLeaseObligations": "balance_sheet",
    "LongTermDebtAndCapitalLeaseObligationsCurrent": "balance_sheet",
    "LongTermDebtAndCapitalLeaseObligationsNoncurrent": "balance_sheet",
    "DebtAndFinanceLeaseObligations": "balance_sheet",
    "DebtAndFinanceLeaseObligationsCurrent": "balance_sheet",
    "DebtAndFinanceLeaseObligationsNoncurrent": "balance_sheet",
    "LongTermDebt": "balance_sheet",
    "LongTermDebtCurrentMaturities": "balance_sheet",
    "LongTermDebtNoncurrent": "balance_sheet",
    "ShortTermBorrowings": "balance_sheet",
    "CommercialPaper": "balance_sheet",
    "NetCashProvidedByUsedInOperatingActivities": "cash_flow",
    "NetCashProvidedByUsedInInvestingActivities": "cash_flow",
    "NetCashProvidedByUsedInFinancingActivities": "cash_flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "cash_flow",
    "DepreciationDepletionAndAmortization": "cash_flow",
    "DepreciationAmortizationAndAccretionNet": "cash_flow",
    "Depreciation": "cash_flow",
    "AmortizationOfIntangibleAssets": "cash_flow",
}


class SECProvider(ProviderBase):
    name = "sec"
    supports_fundamentals = True
    required_dependencies = ("requests",)
    company_tickers_url = "https://www.sec.gov/files/company_tickers.json"
    company_facts_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(self, session: Any | None = None) -> None:
        self._session = session or self._build_session()
        self._company_tickers_cache: dict[str, Any] | None = None

    def _build_session(self) -> Any:
        import requests

        session = requests.Session()
        user_agent = os.getenv(
            "FINTERN_SEC_USER_AGENT",
            "fintern/0.1.0 (configure FINTERN_SEC_USER_AGENT for production use)",
        )
        session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
            }
        )
        return session

    def _get_json(self, url: str) -> dict[str, Any]:
        response = self._session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _load_company_tickers(self) -> dict[str, Any]:
        if self._company_tickers_cache is None:
            self._company_tickers_cache = self._get_json(self.company_tickers_url)

        return self._company_tickers_cache

    def _resolve_cik(self, ticker: str) -> str:
        payload = self._load_company_tickers()

        for entry in payload.values():
            if str(entry.get("ticker", "")).upper() == ticker:
                cik_value = int(entry["cik_str"])
                return f"{cik_value:010d}"

        raise InstrumentResolutionError(
            f"Could not resolve a SEC CIK for ticker `{ticker}`."
        )

    @staticmethod
    def _normalize_company_facts(
        payload: dict[str, Any],
        ticker: str,
        cik: str,
        statements: set[str] | None = None,
    ) -> NormalizedFundamentals:
        rows: list[dict[str, Any]] = []
        facts = payload.get("facts", {})

        for taxonomy, metrics in facts.items():
            if taxonomy != "us-gaap":
                continue

            for metric_name, metric_payload in metrics.items():
                statement = _STATEMENT_BY_METRIC.get(metric_name)

                if statement is None:
                    continue

                if statements is not None and statement not in statements:
                    continue

                label = metric_payload.get("label") or metric_name
                description = metric_payload.get("description")

                for unit, observations in metric_payload.get("units", {}).items():
                    for observation in observations:
                        rows.append(
                            {
                                "ticker": ticker,
                                "cik": cik,
                                "statement": statement,
                                "metric": metric_name,
                                "label": label,
                                "description": description,
                                "unit": unit,
                                "value": observation.get("val"),
                                "period_start": pd.to_datetime(
                                    observation.get("start")
                                ),
                                "period_end": pd.to_datetime(observation.get("end")),
                                "fiscal_year": observation.get("fy"),
                                "fiscal_period": observation.get("fp"),
                                "filed_date": pd.to_datetime(
                                    observation.get("filed")
                                ),
                                "form": observation.get("form"),
                                "frame": observation.get("frame"),
                                "accession_number": observation.get("accn"),
                                "taxonomy": taxonomy,
                                "provider": "sec",
                            }
                        )

        statements_frame = pd.DataFrame(
            rows,
            columns=[
                "ticker",
                "cik",
                "statement",
                "metric",
                "label",
                "description",
                "unit",
                "value",
                "period_start",
                "period_end",
                "fiscal_year",
                "fiscal_period",
                "filed_date",
                "form",
                "frame",
                "accession_number",
                "taxonomy",
                "provider",
            ],
        )
        if not statements_frame.empty:
            statements_frame = statements_frame.drop_duplicates().sort_values(
                ["ticker", "statement", "metric", "period_end", "filed_date"]
            ).reset_index(drop=True)

        profile_frame = pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "cik": cik,
                    "company_name": payload.get("entityName"),
                    "provider": "sec",
                }
            ]
        )

        return {
            "statements": statements_frame,
            "company_profile": profile_frame,
        }

    def download_fundamentals(
        self,
        tickers: Sequence[str],
        statements: Sequence[str] | None = None,
    ) -> NormalizedFundamentals:
        self.ensure_available("fundamentals")
        requested_statements = (
            {statement.lower() for statement in statements}
            if statements
            else None
        )
        statement_frames: list[pd.DataFrame] = []
        profile_frames: list[pd.DataFrame] = []

        for ticker in tickers:
            cik = self._resolve_cik(ticker)
            payload = self._get_json(self.company_facts_url.format(cik=cik))
            normalized = self._normalize_company_facts(
                payload=payload,
                ticker=ticker,
                cik=cik,
                statements=requested_statements,
            )
            statement_frames.append(normalized["statements"])
            profile_frames.append(normalized["company_profile"])

        statements_frame = pd.concat(statement_frames, ignore_index=True, sort=False)
        profile_frame = pd.concat(profile_frames, ignore_index=True, sort=False)

        if not statements_frame.empty:
            statements_frame["ticker"] = (
                statements_frame["ticker"].astype(str).str.upper()
            )

        profile_frame["ticker"] = profile_frame["ticker"].astype(str).str.upper()
        profile_frame = profile_frame.drop_duplicates().reset_index(drop=True)

        return {
            "statements": statements_frame,
            "company_profile": profile_frame,
        }
