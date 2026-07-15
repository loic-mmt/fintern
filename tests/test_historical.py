import numpy as np
import pandas as pd
import pytest

from fintern import HistoricalAnalysis


def _fundamentals() -> dict[str, pd.DataFrame]:
    return {
        "statements": pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "statement": ["income_statement"] * 3,
                "metric": ["Revenues"] * 3,
                "value": [100.0, 120.0, 150.0],
                "period_end": pd.to_datetime(
                    ["2023-12-31", "2024-12-31", "2025-12-31"]
                ),
                "filed_date": pd.to_datetime(
                    ["2024-02-01", "2025-02-01", "2026-02-01"]
                ),
            }
        )
    }


def _analysis() -> HistoricalAnalysis:
    prices = pd.Series(
        [103.0, 101.0, 102.0],
        index=pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"]),
        name="close",
    )
    return HistoricalAnalysis(
        ticker="aapl",
        prices=prices,
        fundamentals=_fundamentals(),
    )


def test_price_history_sorts_and_filters_inclusively() -> None:
    expected = pd.Series(
        [102.0, 103.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
        name="close",
    )

    pd.testing.assert_series_equal(
        _analysis().price_history(start="2025-01-02"),
        expected,
    )


def test_fundamental_history_can_use_reporting_period() -> None:
    expected = pd.Series(
        [120.0, 150.0],
        index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
        name="Revenues",
    )

    pd.testing.assert_series_equal(
        _analysis().fundamental_history(
            "Revenues",
            statement="income_statement",
            start="2024-01-01",
        ),
        expected,
    )


def test_fundamental_history_can_use_publication_date() -> None:
    result = _analysis().fundamental_history("Revenues", date_by="filed_date")

    assert result.index.equals(
        pd.to_datetime(["2024-02-01", "2025-02-01", "2026-02-01"])
    )


def test_analysis_as_of_excludes_future_filings() -> None:
    analysis = HistoricalAnalysis(
        ticker="AAPL",
        fundamentals=_fundamentals(),
        as_of="2025-06-01",
    )

    assert analysis.fundamental_history("Revenues").tolist() == [100.0, 120.0]


def test_history_rejects_reversed_date_range() -> None:
    with pytest.raises(ValueError, match="start must be before"):
        _analysis().price_history(start="2025-02-01", end="2025-01-01")


def test_period_change_returns_fractional_changes() -> None:
    expected = pd.Series(
        [0.2, 0.25],
        index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
        name="Revenues_change",
    )

    pd.testing.assert_series_equal(
        _analysis().period_change("Revenues"),
        expected,
    )


@pytest.mark.parametrize("periods", [0, -1])
def test_period_change_rejects_non_positive_periods(periods: int) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _analysis().period_change("Revenues", periods=periods)


def test_period_change_rejects_zero_prior_value() -> None:
    fundamentals = _fundamentals()
    fundamentals["statements"].loc[0, "value"] = 0.0
    analysis = HistoricalAnalysis(ticker="AAPL", fundamentals=fundamentals)

    with pytest.raises(ValueError, match="prior value is zero"):
        analysis.period_change("Revenues")


def test_compound_annual_growth_rate_uses_actual_dates() -> None:
    elapsed_years = 731 / 365.2425
    expected = (150.0 / 100.0) ** (1 / elapsed_years) - 1

    assert _analysis().compound_annual_growth_rate("Revenues") == pytest.approx(
        expected
    )


def test_compound_annual_growth_rate_rejects_negative_endpoint() -> None:
    fundamentals = _fundamentals()
    fundamentals["statements"].loc[0, "value"] = -100.0
    analysis = HistoricalAnalysis(ticker="AAPL", fundamentals=fundamentals)

    with pytest.raises(ValueError, match="strictly positive endpoints"):
        analysis.compound_annual_growth_rate("Revenues")


def test_trend_returns_normalized_annual_slope() -> None:
    values = pd.Series(
        [100.0, 120.0, 150.0],
        index=pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"]),
    )
    elapsed_years = np.asarray(
        (values.index - values.index[0]).total_seconds(),
        dtype=float,
    ) / (365.2425 * 24 * 60 * 60)
    centered_time = elapsed_years - elapsed_years.mean()
    centered_values = values.to_numpy() - values.mean()
    expected = (
        (centered_time * centered_values).sum()
        / (centered_time**2).sum()
        / values.abs().mean()
    )

    assert _analysis().trend("Revenues") == pytest.approx(expected)


def test_stability_uses_dispersion_of_period_changes() -> None:
    changes = pd.Series([0.2, 0.25])
    expected = 1 / (1 + changes.std(ddof=1))

    assert _analysis().stability("Revenues") == pytest.approx(expected)


def test_align_fundamental_with_prices_uses_backward_filing_date() -> None:
    prices = pd.Series(
        [90.0, 100.0, 110.0, 120.0],
        index=pd.to_datetime(["2024-01-01", "2024-03-01", "2025-03-01", "2026-03-01"]),
    )
    analysis = HistoricalAnalysis(
        ticker="AAPL",
        prices=prices,
        fundamentals=_fundamentals(),
    )
    expected = pd.DataFrame(
        {
            "close": [90.0, 100.0, 110.0, 120.0],
            "value": [float("nan"), 100.0, 120.0, 150.0],
        },
        index=prices.index,
    )

    pd.testing.assert_frame_equal(
        analysis.align_fundamental_with_prices("Revenues"),
        expected,
    )


def test_alignment_ignores_late_amendment_of_an_older_period() -> None:
    statements = _fundamentals()["statements"]
    old_period_amendment = statements.iloc[[0]].assign(
        value=105.0,
        filed_date=pd.Timestamp("2025-04-01"),
    )
    fundamentals = {
        "statements": pd.concat([statements, old_period_amendment], ignore_index=True)
    }
    prices = pd.Series(
        [110.0],
        index=pd.to_datetime(["2025-05-01"]),
    )
    analysis = HistoricalAnalysis(
        ticker="AAPL",
        prices=prices,
        fundamentals=fundamentals,
    )

    result = analysis.align_fundamental_with_prices("Revenues")

    assert result.iloc[0]["value"] == 120.0


def test_snapshot_uses_latest_period_known_as_of_date() -> None:
    snapshot = _analysis().snapshot("2025-06-01")

    assert len(snapshot) == 1
    assert snapshot.loc[0, "value"] == 120.0
    assert snapshot.loc[0, "period_end"] == pd.Timestamp("2024-12-31")
    assert snapshot.loc[0, "filed_date"] == pd.Timestamp("2025-02-01")


def test_snapshot_uses_latest_amendment_for_selected_period() -> None:
    statements = _fundamentals()["statements"]
    amendment = statements.iloc[[1]].assign(
        value=125.0,
        filed_date=pd.Timestamp("2025-03-01"),
    )
    fundamentals = {"statements": pd.concat([statements, amendment])}
    analysis = HistoricalAnalysis(ticker="AAPL", fundamentals=fundamentals)

    snapshot = analysis.snapshot("2025-04-01")

    assert snapshot.loc[0, "value"] == 125.0
    assert snapshot.loc[0, "filed_date"] == pd.Timestamp("2025-03-01")


def test_snapshot_rejects_date_before_first_filing() -> None:
    with pytest.raises(ValueError, match="No fundamentals were available"):
        _analysis().snapshot("2024-01-01")


def test_summary_reports_all_requested_statistics() -> None:
    result = _analysis().summary(["Revenues"])

    assert result.loc[0, "metric"] == "Revenues"
    assert result.loc[0, "latest"] == 150.0
    assert result.loc[0, "prior"] == 120.0
    assert result.loc[0, "change"] == pytest.approx(0.25)
    assert result.loc[0, "cagr"] == pytest.approx(
        _analysis().compound_annual_growth_rate("Revenues")
    )
    assert result.loc[0, "observations"] == 3
