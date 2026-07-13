import pandas as pd
import pytest

from fintern import Growth


def _sample_prices() -> pd.Series:
    return pd.Series(
        [100.0, 120.0, 150.0],
        index=pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]),
    )


def _sample_fundamentals() -> dict[str, pd.DataFrame]:
    periods = pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"])
    metrics = {
        "Revenues": ("income_statement", [100.0, 120.0, 150.0]),
        "NetIncomeLoss": ("income_statement", [10.0, 12.0, 18.0]),
        "EarningsPerShareDiluted": ("income_statement", [1.0, 1.2, 1.5]),
        "NetCashProvidedByUsedInOperatingActivities": (
            "cash_flow",
            [20.0, 25.0, 35.0],
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": (
            "cash_flow",
            [5.0, 5.0, 8.0],
        ),
        "StockholdersEquity": ("balance_sheet", [50.0, 60.0, 75.0]),
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


def _growth() -> Growth:
    return Growth(
        ticker="aapl",
        prices=_sample_prices(),
        fundamentals=_sample_fundamentals(),
    )


def test_price_momentum_uses_requested_trailing_periods() -> None:
    assert _growth().price_momentum(periods=2) == pytest.approx(0.5)


def test_rolling_price_momentum_returns_series() -> None:
    expected = pd.Series(
        [0.5],
        index=pd.to_datetime(["2025-09-30"]),
        name="price_momentum",
    )

    pd.testing.assert_series_equal(
        _growth().rolling_price_momentum(periods=2),
        expected,
    )


def test_annualized_price_growth_uses_elapsed_intervals() -> None:
    expected = (150.0 / 100.0) ** (4.0 / 2.0) - 1

    assert _growth().annualized_price_growth(periods_per_year=4) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("method_name", "expected_values", "series_name"),
    [
        ("revenue_growth", [0.2, 0.25], "revenue_growth"),
        ("net_income_growth", [0.2, 0.5], "net_income_growth"),
        (
            "earnings_per_share_growth",
            [0.2, 0.25],
            "earnings_per_share_growth",
        ),
        ("book_value_growth", [0.2, 0.25], "book_value_growth"),
    ],
)
def test_fundamental_growth_metrics(
    method_name: str,
    expected_values: list[float],
    series_name: str,
) -> None:
    expected = pd.Series(
        expected_values,
        index=pd.to_datetime(["2025-06-30", "2025-09-30"]),
        name=series_name,
    )

    result = getattr(_growth(), method_name)()

    pd.testing.assert_series_equal(result, expected)


def test_free_cash_flow_growth_uses_operating_cash_flow_minus_capex() -> None:
    expected = pd.Series(
        [20.0 / 15.0 - 1, 27.0 / 20.0 - 1],
        index=pd.to_datetime(["2025-06-30", "2025-09-30"]),
        name="free_cash_flow_growth",
    )

    pd.testing.assert_series_equal(_growth().free_cash_flow_growth(), expected)


@pytest.mark.parametrize("periods", [0, -1])
def test_growth_rejects_non_positive_periods(periods: int) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _growth().price_momentum(periods=periods)


def test_growth_rejects_window_larger_than_history() -> None:
    with pytest.raises(ValueError, match="smaller than the number of prices"):
        _growth().price_momentum(periods=3)
