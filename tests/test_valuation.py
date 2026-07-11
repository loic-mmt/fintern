import pandas as pd
import pytest

from fintern.metrics.valuation import Valuation


def _sample_prices() -> pd.Series:
    return pd.Series(
        [6.0, 7.0, 8.0],
        index=pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]),
        name="AAPL",
    )


def _sample_fundamentals_with_total_debt() -> dict[str, pd.DataFrame]:
    statements = pd.DataFrame(
        {
            "ticker": [
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
            ],
            "statement": [
                "balance_sheet",
                "income_statement",
                "income_statement",
                "income_statement",
                "balance_sheet",
                "balance_sheet",
                "cash_flow",
                "cash_flow",
                "income_statement",
                "cash_flow",
            ],
            "metric": [
                "CommonStockSharesOutstanding",
                "Revenues",
                "NetIncomeLoss",
                "EarningsPerShareDiluted",
                "StockholdersEquity",
                "LongTermDebtAndCapitalLeaseObligations",
                "NetCashProvidedByUsedInOperatingActivities",
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "OperatingIncomeLoss",
                "DepreciationDepletionAndAmortization",
            ],
            "value": [10.0, 50.0, 20.0, 2.0, 40.0, 15.0, 12.0, 3.0, 18.0, 4.0],
            "period_end": pd.to_datetime(["2025-09-30"] * 10),
            "filed_date": pd.to_datetime(["2025-11-01"] * 10),
            "provider": ["sec"] * 10,
        }
    )
    return {"statements": statements, "company_profile": pd.DataFrame()}


def _sample_fundamentals_with_debt_components() -> dict[str, pd.DataFrame]:
    statements = pd.DataFrame(
        {
            "ticker": [
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
            ],
            "statement": [
                "balance_sheet",
                "balance_sheet",
                "balance_sheet",
                "balance_sheet",
                "income_statement",
            ],
            "metric": [
                "CommonStockSharesOutstanding",
                "LongTermDebtAndCapitalLeaseObligationsCurrent",
                "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
                "CashAndCashEquivalentsAtCarryingValue",
                "Revenues",
            ],
            "value": [10.0, 4.0, 11.0, 5.0, 50.0],
            "period_end": pd.to_datetime(["2025-09-30"] * 5),
            "filed_date": pd.to_datetime(["2025-11-01"] * 5),
            "provider": ["sec"] * 5,
        }
    )
    return {"statements": statements, "company_profile": pd.DataFrame()}


def test_market_cap_uses_latest_close_and_shares() -> None:
    valuation = Valuation(
        ticker="AAPL",
        prices=_sample_prices(),
        fundamentals=_sample_fundamentals_with_total_debt(),
    )

    assert valuation.market_cap() == pytest.approx(80.0)


def test_price_to_sales_uses_market_cap_over_revenue() -> None:
    valuation = Valuation(
        ticker="AAPL",
        prices=_sample_prices(),
        fundamentals=_sample_fundamentals_with_total_debt(),
    )

    assert valuation.price_to_sales() == pytest.approx(1.6)


def test_price_to_earnings_uses_market_cap_over_net_income() -> None:
    valuation = Valuation(
        ticker="AAPL",
        prices=_sample_prices(),
        fundamentals=_sample_fundamentals_with_total_debt(),
    )

    assert valuation.price_to_earnings() == pytest.approx(4.0)


def test_price_to_book_uses_market_cap_over_equity() -> None:
    valuation = Valuation(
        ticker="AAPL",
        prices=_sample_prices(),
        fundamentals=_sample_fundamentals_with_total_debt(),
    )

    assert valuation.price_to_book() == pytest.approx(2.0)


def test_enterprise_value_uses_direct_total_debt_and_cash() -> None:
    fundamentals = _sample_fundamentals_with_total_debt()
    fundamentals["statements"] = pd.concat(
        [
            fundamentals["statements"],
            pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "statement": ["balance_sheet"],
                    "metric": ["CashAndCashEquivalentsAtCarryingValue"],
                    "value": [5.0],
                    "period_end": pd.to_datetime(["2025-09-30"]),
                    "filed_date": pd.to_datetime(["2025-11-01"]),
                    "provider": ["sec"],
                }
            ),
        ],
        ignore_index=True,
    )
    valuation = Valuation(
        ticker="AAPL",
        prices=_sample_prices(),
        fundamentals=fundamentals,
    )

    assert valuation.enterprise_value() == pytest.approx(90.0)
    assert valuation.ev_to_sales() == pytest.approx(1.8)


def test_enterprise_value_falls_back_to_current_plus_noncurrent_debt() -> None:
    valuation = Valuation(
        ticker="AAPL",
        prices=_sample_prices(),
        fundamentals=_sample_fundamentals_with_debt_components(),
    )

    assert valuation.enterprise_value() == pytest.approx(90.0)


def test_free_cash_flow_yield_uses_operating_cash_flow_minus_capex() -> None:
    fundamentals = _sample_fundamentals_with_total_debt()
    fundamentals["statements"] = pd.concat(
        [
            fundamentals["statements"],
            pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "statement": ["balance_sheet"],
                    "metric": ["CashAndCashEquivalentsAtCarryingValue"],
                    "value": [5.0],
                    "period_end": pd.to_datetime(["2025-09-30"]),
                    "filed_date": pd.to_datetime(["2025-11-01"]),
                    "provider": ["sec"],
                }
            ),
        ],
        ignore_index=True,
    )
    valuation = Valuation(
        ticker="AAPL",
        prices=_sample_prices(),
        fundamentals=fundamentals,
    )

    assert valuation.free_cash_flow_yield() == pytest.approx(9.0 / 80.0)


def test_latest_ebitda_uses_operating_income_plus_d_and_a() -> None:
    valuation = Valuation(
        ticker="AAPL",
        prices=_sample_prices(),
        fundamentals=_sample_fundamentals_with_total_debt(),
    )

    assert valuation._latest_ebitda() == pytest.approx(22.0)


def test_ev_to_ebitda_uses_enterprise_value_over_latest_ebitda() -> None:
    fundamentals = _sample_fundamentals_with_total_debt()
    fundamentals["statements"] = pd.concat(
        [
            fundamentals["statements"],
            pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "statement": ["balance_sheet"],
                    "metric": ["CashAndCashEquivalentsAtCarryingValue"],
                    "value": [5.0],
                    "period_end": pd.to_datetime(["2025-09-30"]),
                    "filed_date": pd.to_datetime(["2025-11-01"]),
                    "provider": ["sec"],
                }
            ),
        ],
        ignore_index=True,
    )
    valuation = Valuation(
        ticker="AAPL",
        prices=_sample_prices(),
        fundamentals=fundamentals,
    )

    assert valuation.ev_to_ebitda() == pytest.approx(90.0 / 22.0)
