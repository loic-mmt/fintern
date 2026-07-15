import pandas as pd
import pytest

from fintern import RedFlag, RedFlagAnalysis, RedFlagThresholds


def _fundamentals() -> dict[str, pd.DataFrame]:
    return {
        "statements": pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "statement": ["income_statement", "income_statement"],
                "metric": ["Revenues", "Revenues"],
                "value": [100.0, 120.0],
                "period_end": pd.to_datetime(["2024-12-31", "2025-12-31"]),
                "filed_date": pd.to_datetime(["2025-02-01", "2026-02-01"]),
            }
        )
    }


def _complete_fundamentals() -> dict[str, pd.DataFrame]:
    periods = pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"])
    metrics = {
        "Revenues": ("income_statement", [120.0, 110.0, 90.0]),
        "NetIncomeLoss": ("income_statement", [24.0, 16.0, 9.0]),
        "OperatingIncomeLoss": ("income_statement", [30.0, 18.0, 7.0]),
        "NetCashProvidedByUsedInOperatingActivities": (
            "cash_flow",
            [40.0, 25.0, 5.0],
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": (
            "cash_flow",
            [10.0, 15.0, 12.0],
        ),
        "LongTermDebt": ("balance_sheet", [40.0, 60.0, 80.0]),
        "StockholdersEquity": ("balance_sheet", [100.0, 90.0, 80.0]),
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
                    "filed_date": period + pd.Timedelta(days=45),
                }
            )

    return {"statements": pd.DataFrame(rows)}


def _complete_analysis() -> RedFlagAnalysis:
    prices = pd.Series(
        [100.0, 140.0, 80.0, 120.0, 60.0],
        index=pd.date_range("2026-03-02", periods=5, freq="B"),
    )
    return RedFlagAnalysis(
        ticker="AAPL",
        prices=prices,
        fundamentals=_complete_fundamentals(),
    )


def test_red_flag_normalizes_fields_and_dates() -> None:
    flag = RedFlag(
        code="Declining Revenue",
        severity="medium",
        message="  Revenue declined for two periods.  ",
        metric="Revenues",
        value=-0.10,
        threshold=0.0,
        period_start="2024-12-31",
        period_end="2025-12-31",
    )

    assert flag.code == "declining_revenue"
    assert flag.message == "Revenue declined for two periods."
    assert flag.period_start == pd.Timestamp("2024-12-31")
    assert flag.period_end == pd.Timestamp("2025-12-31")


def test_red_flag_rejects_invalid_period() -> None:
    with pytest.raises(ValueError, match="period_start"):
        RedFlag(
            code="invalid_period",
            severity="low",
            message="Invalid period.",
            period_start="2025-12-31",
            period_end="2024-12-31",
        )


def test_thresholds_validate_drawdown_range() -> None:
    with pytest.raises(ValueError, match="between -1 and 0"):
        RedFlagThresholds(maximum_drawdown=0.10)


def test_analysis_builds_historical_view_from_same_inputs() -> None:
    analysis = RedFlagAnalysis(ticker="aapl", fundamentals=_fundamentals())

    history = analysis._historical().fundamental_history("Revenues")

    assert analysis.ticker == "AAPL"
    assert history.tolist() == [100.0, 120.0]


def test_analysis_propagates_as_of_to_historical_view() -> None:
    analysis = RedFlagAnalysis(
        ticker="AAPL",
        fundamentals=_fundamentals(),
        as_of="2025-06-01",
    )

    assert analysis._historical().fundamental_history("Revenues").tolist() == [100.0]


def test_to_frame_preserves_stable_schema() -> None:
    flag = RedFlag(
        code="declining_revenue",
        severity="medium",
        message="Revenue declined.",
        metric="Revenues",
        value=-0.1,
    )

    result = RedFlagAnalysis.to_frame([flag])

    assert result.columns.tolist() == [
        "code",
        "severity",
        "message",
        "metric",
        "value",
        "reference_value",
        "threshold",
        "period_start",
        "period_end",
    ]
    assert result.loc[0, "code"] == "declining_revenue"


def test_empty_flag_list_returns_empty_frame_with_schema() -> None:
    result = RedFlagAnalysis.to_frame([])

    assert result.empty
    assert "severity" in result.columns


def test_declining_revenue_detects_consecutive_declines() -> None:
    flags = _complete_analysis().declining_revenue()

    assert len(flags) == 1
    assert flags[0].code == "declining_revenue"
    assert flags[0].severity == "high"
    assert flags[0].value == pytest.approx(90.0 / 120.0 - 1)


def test_declining_earnings_uses_net_income() -> None:
    flag = _complete_analysis().declining_earnings()[0]

    assert flag.code == "declining_earnings"
    assert flag.severity == "medium"
    assert flag.metric == "NetIncomeLoss"
    assert flag.value == 9.0
    assert flag.reference_value == 16.0


def test_declining_earnings_falls_back_to_eps() -> None:
    fundamentals = _complete_fundamentals()
    statements = fundamentals["statements"]
    statements = statements.loc[statements["metric"] != "NetIncomeLoss"].copy()
    periods = pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"])
    eps_rows = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 3,
            "statement": ["income_statement"] * 3,
            "metric": ["EarningsPerShareDiluted"] * 3,
            "value": [3.0, 2.0, 1.0],
            "period_end": periods,
            "filed_date": periods + pd.Timedelta(days=45),
        }
    )
    fundamentals["statements"] = pd.concat(
        [statements, eps_rows],
        ignore_index=True,
    )

    flag = RedFlagAnalysis(
        ticker="AAPL",
        fundamentals=fundamentals,
    ).declining_earnings()[0]

    assert flag.metric == "EarningsPerShareDiluted"


def test_negative_free_cash_flow_distinguishes_latest_period() -> None:
    flag = _complete_analysis().negative_free_cash_flow()[0]

    assert flag.code == "negative_free_cash_flow"
    assert flag.severity == "medium"
    assert flag.value == -7.0
    assert flag.reference_value == 10.0


def test_margin_deterioration_returns_one_flag_per_margin() -> None:
    flags = _complete_analysis().margin_deterioration()

    assert [flag.code for flag in flags] == [
        "operating_margin_deterioration",
        "net_margin_deterioration",
    ]
    assert all(flag.severity == "high" for flag in flags)


def test_earnings_cash_flow_divergence_compares_latest_growth() -> None:
    flag = _complete_analysis().earnings_cash_flow_divergence()[0]

    assert flag.code == "earnings_cash_flow_divergence"
    assert flag.severity == "high"
    assert flag.value == pytest.approx(9.0 / 16.0 - 1)
    assert flag.reference_value == pytest.approx(5.0 / 25.0 - 1)


def test_rising_leverage_uses_debt_to_equity() -> None:
    flag = _complete_analysis().rising_leverage()[0]

    assert flag.code == "rising_leverage"
    assert flag.severity == "high"
    assert flag.value == pytest.approx(1.0)
    assert flag.reference_value == pytest.approx(60.0 / 90.0)


def test_rising_leverage_builds_debt_from_components() -> None:
    fundamentals = _complete_fundamentals()
    statements = fundamentals["statements"]
    debt_rows = statements.loc[statements["metric"] == "LongTermDebt"].copy()
    statements = statements.loc[statements["metric"] != "LongTermDebt"].copy()
    current_debt = debt_rows.assign(
        metric="LongTermDebtCurrentMaturities",
        value=[10.0, 20.0, 30.0],
    )
    noncurrent_debt = debt_rows.assign(
        metric="LongTermDebtNoncurrent",
        value=[30.0, 40.0, 50.0],
    )
    fundamentals["statements"] = pd.concat(
        [statements, current_debt, noncurrent_debt],
        ignore_index=True,
    )

    flag = RedFlagAnalysis(
        ticker="AAPL",
        fundamentals=fundamentals,
    ).rising_leverage()[0]

    assert flag.value == pytest.approx(1.0)


def test_non_positive_equity_is_critical() -> None:
    fundamentals = _complete_fundamentals()
    statements = fundamentals["statements"]
    latest_equity = (statements["metric"] == "StockholdersEquity") & (
        statements["period_end"] == pd.Timestamp("2025-12-31")
    )
    statements.loc[latest_equity, "value"] = -5.0
    analysis = RedFlagAnalysis(ticker="AAPL", fundamentals=fundamentals)

    flag = analysis.rising_leverage()[0]

    assert flag.code == "non_positive_equity"
    assert flag.severity == "critical"


def test_high_market_risk_emits_volatility_and_drawdown_flags() -> None:
    flags = _complete_analysis().high_market_risk()

    assert {flag.code for flag in flags} == {
        "high_annualized_volatility",
        "severe_drawdown",
    }


def test_calm_market_prices_do_not_raise_risk_flags() -> None:
    prices = pd.Series(
        [100.0, 100.1, 100.2, 100.3, 100.4],
        index=pd.date_range("2026-03-02", periods=5, freq="B"),
    )
    analysis = RedFlagAnalysis(ticker="AAPL", prices=prices)

    assert analysis.high_market_risk() == []


def test_stale_fundamentals_uses_latest_filing_before_as_of() -> None:
    flag = _complete_analysis().stale_fundamentals("2027-03-01")[0]

    assert flag.code == "stale_fundamentals"
    assert flag.severity == "medium"
    assert flag.value > 365


def test_recent_fundamentals_do_not_raise_flag() -> None:
    assert _complete_analysis().stale_fundamentals("2026-03-01") == []


def test_run_all_sorts_flags_by_severity() -> None:
    flags = _complete_analysis().run_all()
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    ranks = [severity_rank[flag.severity] for flag in flags]

    assert flags
    assert ranks == sorted(ranks)


def test_run_all_strict_propagates_missing_data() -> None:
    analysis = RedFlagAnalysis(ticker="AAPL", fundamentals=_fundamentals())

    with pytest.raises(ValueError, match="declining_revenue"):
        analysis.run_all(strict=True)
