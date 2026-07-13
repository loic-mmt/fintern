import pandas as pd
import pytest

from fintern import Profitability


def _sample_fundamentals() -> dict[str, pd.DataFrame]:
    periods = pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"])
    metrics = {
        "Revenues": ("income_statement", [100.0, 120.0, 150.0]),
        "GrossProfit": ("income_statement", [40.0, 50.0, 66.0]),
        "OperatingIncomeLoss": ("income_statement", [20.0, 24.0, 33.0]),
        "NetIncomeLoss": ("income_statement", [10.0, 12.0, 18.0]),
        "IncomeTaxExpenseBenefit": ("income_statement", [2.5, 3.0, 4.5]),
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
            "ExtraordinaryItemsNoncontrollingInterest"
        ): (
            "income_statement",
            [12.5, 15.0, 22.5],
        ),
        "NetCashProvidedByUsedInOperatingActivities": (
            "cash_flow",
            [20.0, 25.0, 35.0],
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": (
            "cash_flow",
            [5.0, 5.0, 8.0],
        ),
        "Assets": ("balance_sheet", [100.0, 120.0, 150.0]),
        "StockholdersEquity": ("balance_sheet", [50.0, 60.0, 75.0]),
        "LongTermDebtAndCapitalLeaseObligations": (
            "balance_sheet",
            [20.0, 24.0, 30.0],
        ),
        "CashAndCashEquivalentsAtCarryingValue": (
            "balance_sheet",
            [5.0, 6.0, 8.0],
        ),
    }
    rows = []

    for metric, (statement, values) in metrics.items():
        for period, value in zip(periods, values, strict=True):
            rows.append(
                {
                    "ticker": "AAPL",
                    "statement": statement,
                    "metric": metric,
                    "value": value,
                    "period_end": period,
                    "filed_date": period + pd.Timedelta(days=30),
                }
            )

    return {
        "statements": pd.DataFrame(rows),
        "company_profile": pd.DataFrame(),
    }


def _profitability(
    fundamentals: dict[str, pd.DataFrame] | None = None,
) -> Profitability:
    return Profitability(
        ticker="aapl",
        fundamentals=fundamentals or _sample_fundamentals(),
    )


@pytest.mark.parametrize(
    ("method_name", "expected_values", "series_name"),
    [
        ("gross_margin", [0.4, 50.0 / 120.0, 0.44], "gross_margin"),
        ("operating_margin", [0.2, 0.2, 0.22], "operating_margin"),
        ("net_margin", [0.1, 0.1, 0.12], "net_margin"),
        ("ebit_margin", [0.2, 0.2, 0.22], "ebit_margin"),
        (
            "free_cash_flow_margin",
            [0.15, 20.0 / 120.0, 0.18],
            "free_cash_flow_margin",
        ),
    ],
)
def test_margin_metrics(
    method_name: str,
    expected_values: list[float],
    series_name: str,
) -> None:
    expected = pd.Series(
        expected_values,
        index=pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]),
        name=series_name,
    )

    result = getattr(_profitability(), method_name)()

    pd.testing.assert_series_equal(result, expected)


def test_gross_margin_falls_back_to_revenue_minus_cost_of_revenue() -> None:
    fundamentals = _sample_fundamentals()
    statements = fundamentals["statements"]
    statements = statements.loc[statements["metric"] != "GrossProfit"].copy()
    cost_rows = statements.loc[statements["metric"] == "Revenues"].copy()
    cost_rows["metric"] = "CostOfGoodsSold"
    cost_rows["value"] = [60.0, 70.0, 84.0]
    fundamentals["statements"] = pd.concat(
        [statements, cost_rows],
        ignore_index=True,
    )
    expected = pd.Series(
        [0.4, 50.0 / 120.0, 0.44],
        index=pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]),
        name="gross_margin",
    )

    pd.testing.assert_series_equal(
        _profitability(fundamentals).gross_margin(),
        expected,
    )


def test_return_on_assets_uses_average_assets() -> None:
    expected = pd.Series(
        [12.0 / 110.0, 18.0 / 135.0],
        index=pd.to_datetime(["2025-06-30", "2025-09-30"]),
        name="return_on_assets",
    )

    pd.testing.assert_series_equal(_profitability().return_on_assets(), expected)


def test_return_on_equity_uses_average_equity() -> None:
    expected = pd.Series(
        [12.0 / 55.0, 18.0 / 67.5],
        index=pd.to_datetime(["2025-06-30", "2025-09-30"]),
        name="return_on_equity",
    )

    pd.testing.assert_series_equal(_profitability().return_on_equity(), expected)


def test_roic_uses_nopat_and_average_invested_capital() -> None:
    expected = pd.Series(
        [24.0 * 0.8 / 71.5, 33.0 * 0.8 / 87.5],
        index=pd.to_datetime(["2025-06-30", "2025-09-30"]),
        name="return_on_invested_capital",
    )

    pd.testing.assert_series_equal(
        _profitability().return_on_invested_capital(),
        expected,
    )


def test_roic_accepts_explicit_tax_rate() -> None:
    result = _profitability().return_on_invested_capital(tax_rate=0.25)

    assert result.iloc[-1] == pytest.approx(33.0 * 0.75 / 87.5)


@pytest.mark.parametrize("tax_rate", [-0.1, 1.0])
def test_roic_rejects_invalid_explicit_tax_rate(tax_rate: float) -> None:
    with pytest.raises(ValueError, match="tax_rate"):
        _profitability().return_on_invested_capital(tax_rate=tax_rate)
