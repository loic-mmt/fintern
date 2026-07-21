import numpy as np
import pandas as pd
import pytest

from fintern import (
    CompanyScore,
    RedFlag,
    ScoreComponent,
    ScoringAnalysis,
    ScoringConfig,
)


def _fundamentals() -> dict[str, pd.DataFrame]:
    periods = pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"])
    metrics = {
        "Revenues": ("income_statement", [100.0, 110.0, 120.0]),
        "OperatingIncomeLoss": ("income_statement", [15.0, 18.0, 21.0]),
        "NetIncomeLoss": ("income_statement", [10.0, 12.0, 14.0]),
        "IncomeTaxExpenseBenefit": ("income_statement", [3.0, 3.6, 4.2]),
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
            "ExtraordinaryItemsNoncontrollingInterest"
        ): ("income_statement", [13.0, 15.6, 18.2]),
        "EarningsBeforeInterestTaxesDepreciationAndAmortization": (
            "income_statement",
            [18.0, 21.0, 24.0],
        ),
        "NetCashProvidedByUsedInOperatingActivities": (
            "cash_flow",
            [15.0, 17.0, 20.0],
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": (
            "cash_flow",
            [5.0, 5.0, 6.0],
        ),
        "StockholdersEquity": ("balance_sheet", [60.0, 66.0, 72.0]),
        "LongTermDebt": ("balance_sheet", [20.0, 20.0, 20.0]),
        "CashAndCashEquivalentsAtCarryingValue": (
            "balance_sheet",
            [10.0, 11.0, 12.0],
        ),
        "CommonStockSharesOutstanding": (
            "balance_sheet",
            [5.0, 5.0, 5.0],
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

    return {"statements": pd.DataFrame(rows)}


def _prices() -> pd.Series:
    index = pd.date_range("2024-01-02", "2026-03-31", freq="B")
    return pd.Series(np.linspace(100.0, 130.0, len(index)), index=index)


def test_scoring_config_rejects_decreasing_penalties() -> None:
    with pytest.raises(ValueError, match="increase with severity"):
        ScoringConfig(medium_flag_penalty=0.5)


def test_score_component_preserves_auditable_contribution() -> None:
    component = ScoreComponent(
        category="growth",
        metric="revenue_growth",
        value=0.12,
        score=80.0,
        weight=0.20,
        explanation="Revenue grew by 12%.",
    )

    assert component.contribution == pytest.approx(16.0)
    assert component.as_dict()["status"] == "scored"


def test_linear_score_supports_both_directions_and_clips() -> None:
    assert ScoringAnalysis.linear_score(0.5, 0.0, 1.0) == 50.0
    assert ScoringAnalysis.linear_score(0.5, 0.0, 1.0, "lower_is_better") == 50.0
    assert ScoringAnalysis.linear_score(2.0, 0.0, 1.0) == 100.0
    assert ScoringAnalysis.linear_score(2.0, 0.0, 1.0, "lower_is_better") == 0.0


def test_aggregate_renormalizes_coverage_and_applies_flag_penalty() -> None:
    analysis = ScoringAnalysis(ticker="aapl")
    components = [
        ScoreComponent(
            category="growth",
            metric="revenue_growth",
            value=0.10,
            score=80.0,
            weight=0.60,
            explanation="Available.",
        ),
        analysis.missing_component(
            category="valuation",
            metric="price_to_earnings",
            weight=0.40,
            reason="Unavailable.",
        ),
    ]
    flags = [RedFlag(code="warning", severity="high", message="Material warning.")]

    result = analysis.aggregate(components, flags)

    assert isinstance(result, CompanyScore)
    assert result.coverage == pytest.approx(0.60)
    assert result.penalty == 7.0
    assert result.total_score == pytest.approx(73.0)
    assert result.grade == "B"


def test_aggregate_withholds_score_below_minimum_coverage() -> None:
    analysis = ScoringAnalysis(ticker="AAPL")
    components = [
        ScoreComponent(
            category="risk",
            metric="maximum_drawdown",
            value=-0.10,
            score=80.0,
            weight=0.50,
            explanation="Available.",
        ),
        analysis.missing_component(
            category="valuation",
            metric="price_to_earnings",
            weight=0.50,
            reason="Unavailable.",
        ),
    ]

    result = analysis.aggregate(components, [])

    assert result.total_score is None
    assert result.grade is None
    assert result.coverage == 0.50


def test_full_scoring_is_explainable_and_complete() -> None:
    result = ScoringAnalysis(
        ticker="aapl",
        prices=_prices(),
        fundamentals=_fundamentals(),
        as_of="2026-03-31",
    ).score()

    assert result.ticker == "AAPL"
    assert result.total_score is not None
    assert result.grade in {"A", "B", "C", "D", "E"}
    assert result.coverage == 1.0
    assert len(result.components) == 17
    assert all(component.status == "scored" for component in result.components)
    assert result.red_flags == ()
    assert result.component_frame().columns.tolist() == [
        "category",
        "metric",
        "value",
        "score",
        "weight",
        "explanation",
        "status",
        "contribution",
    ]


def test_scoring_without_inputs_reports_missing_coverage() -> None:
    result = ScoringAnalysis(ticker="AAPL").score()

    assert result.total_score is None
    assert result.grade is None
    assert result.coverage == 0.0
    assert len(result.components) == 17
    assert all(component.status == "missing" for component in result.components)


def test_as_of_excludes_later_prices() -> None:
    analysis = ScoringAnalysis(
        ticker="AAPL",
        prices=_prices(),
        as_of="2025-03-31",
    )

    assert analysis._scoring_prices().index.max() == pd.Timestamp("2025-03-31")
