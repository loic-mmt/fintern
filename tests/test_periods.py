import pandas as pd
import pytest

from fintern import (
    Growth,
    build_ttm_fundamentals,
    classify_fundamental_periods,
    select_fundamental_periods,
)


def _fact(
    *,
    statement: str,
    metric: str,
    value: float,
    period_end: str,
    filed_date: str,
    fiscal_period: str,
    form: str,
    period_start: str | None = None,
    frame: str | None = None,
    accession_number: str,
) -> dict[str, object]:
    return {
        "ticker": "AAPL",
        "statement": statement,
        "metric": metric,
        "unit": "USD",
        "value": value,
        "period_start": period_start,
        "period_end": period_end,
        "filed_date": filed_date,
        "fiscal_year": 2024,
        "fiscal_period": fiscal_period,
        "form": form,
        "frame": frame,
        "accession_number": accession_number,
        "taxonomy": "us-gaap",
        "provider": "sec",
    }


def _sample_statements(include_direct_q2: bool = False) -> pd.DataFrame:
    rows = [
        _fact(
            statement="income_statement",
            metric="Revenues",
            value=100.0,
            period_start="2024-01-01",
            period_end="2024-03-31",
            filed_date="2024-04-20",
            fiscal_period="Q1",
            form="10-Q",
            frame="CY2024Q1",
            accession_number="q1-original",
        ),
        _fact(
            statement="income_statement",
            metric="Revenues",
            value=110.0,
            period_start="2024-01-01",
            period_end="2024-03-31",
            filed_date="2024-05-15",
            fiscal_period="Q1",
            form="10-Q/A",
            frame="CY2024Q1",
            accession_number="q1-amended",
        ),
        _fact(
            statement="income_statement",
            metric="Revenues",
            value=250.0,
            period_start="2024-01-01",
            period_end="2024-06-30",
            filed_date="2024-08-01",
            fiscal_period="Q2",
            form="10-Q",
            accession_number="q2-ytd",
        ),
        _fact(
            statement="income_statement",
            metric="Revenues",
            value=420.0,
            period_start="2024-01-01",
            period_end="2024-09-30",
            filed_date="2024-11-01",
            fiscal_period="Q3",
            form="10-Q",
            accession_number="q3-ytd",
        ),
        _fact(
            statement="income_statement",
            metric="Revenues",
            value=650.0,
            period_start="2024-01-01",
            period_end="2024-12-31",
            filed_date="2025-02-15",
            fiscal_period="FY",
            form="10-K",
            accession_number="fy",
        ),
        _fact(
            statement="balance_sheet",
            metric="Assets",
            value=900.0,
            period_end="2024-03-31",
            filed_date="2024-04-20",
            fiscal_period="Q1",
            form="10-Q",
            accession_number="assets-q1",
        ),
        _fact(
            statement="balance_sheet",
            metric="Assets",
            value=950.0,
            period_end="2024-06-30",
            filed_date="2024-08-01",
            fiscal_period="Q2",
            form="10-Q",
            accession_number="assets-q2",
        ),
        _fact(
            statement="balance_sheet",
            metric="Assets",
            value=980.0,
            period_end="2024-09-30",
            filed_date="2024-11-01",
            fiscal_period="Q3",
            form="10-Q",
            accession_number="assets-q3",
        ),
        _fact(
            statement="balance_sheet",
            metric="Assets",
            value=1000.0,
            period_end="2024-12-31",
            filed_date="2025-02-15",
            fiscal_period="FY",
            form="10-K",
            accession_number="assets-fy",
        ),
    ]

    if include_direct_q2:
        rows.append(
            _fact(
                statement="income_statement",
                metric="Revenues",
                value=140.0,
                period_start="2024-04-01",
                period_end="2024-06-30",
                filed_date="2024-08-01",
                fiscal_period="Q2",
                form="10-Q",
                frame="CY2024Q2",
                accession_number="q2-direct",
            )
        )

    return pd.DataFrame(rows)


def test_classify_fundamental_periods_distinguishes_fact_semantics() -> None:
    classified = classify_fundamental_periods(_sample_statements())

    revenue_types = set(
        classified.loc[classified["metric"] == "Revenues", "period_type"]
    )
    asset_types = set(classified.loc[classified["metric"] == "Assets", "period_type"])

    assert revenue_types == {"quarterly", "ytd", "annual"}
    assert asset_types == {"instant"}


def test_quarterly_selection_uses_latest_amendment_and_derives_quarters() -> None:
    selected = select_fundamental_periods(
        _sample_statements(),
        frequency="quarterly",
    )
    revenue = selected.loc[selected["metric"] == "Revenues"].sort_values("period_end")

    assert revenue["fiscal_period"].tolist() == ["Q1", "Q2", "Q3", "Q4"]
    assert revenue["value"].tolist() == pytest.approx([110.0, 140.0, 170.0, 230.0])
    assert revenue["is_derived"].tolist() == [False, True, True, True]


def test_quarterly_selection_prefers_reported_discrete_quarter() -> None:
    selected = select_fundamental_periods(
        _sample_statements(include_direct_q2=True),
        frequency="quarterly",
    )
    q2 = selected.loc[
        (selected["metric"] == "Revenues") & (selected["fiscal_period"] == "Q2")
    ]

    assert len(q2) == 1
    assert q2.iloc[0]["value"] == pytest.approx(140.0)
    assert not bool(q2.iloc[0]["is_derived"])


def test_as_of_prevents_future_amendment_lookahead() -> None:
    selected = select_fundamental_periods(
        _sample_statements(),
        frequency="quarterly",
        as_of="2024-04-30",
    )
    revenue = selected.loc[selected["metric"] == "Revenues"]

    assert revenue["value"].tolist() == [100.0]
    assert revenue["filed_date"].max() <= pd.Timestamp("2024-04-30")


def test_annual_selection_keeps_annual_flow_and_year_end_balance() -> None:
    selected = select_fundamental_periods(
        _sample_statements(),
        frequency="annual",
    )

    assert set(selected["metric"]) == {"Revenues", "Assets"}
    assert selected.loc[selected["metric"] == "Revenues", "value"].item() == 650.0
    assert selected.loc[selected["metric"] == "Assets", "value"].item() == 1000.0


def test_build_ttm_fundamentals_sums_four_quarters_and_keeps_latest_balance() -> None:
    result = build_ttm_fundamentals(_sample_statements())
    revenue = result.loc[result["metric"] == "Revenues"].iloc[0]
    assets = result.loc[result["metric"] == "Assets"].iloc[0]

    assert revenue["period_type"] == "ttm"
    assert revenue["fiscal_period"] == "TTM"
    assert revenue["value"] == pytest.approx(650.0)
    assert revenue["derivation"] == "sum_four_consecutive_quarters"
    assert assets["value"] == pytest.approx(1000.0)
    assert assets["period_type"] == "instant"


def test_build_ttm_as_of_returns_only_information_available_at_cutoff() -> None:
    result = build_ttm_fundamentals(
        _sample_statements(),
        as_of="2024-11-15",
    )

    assert "Revenues" not in set(result["metric"])
    assert result.loc[result["metric"] == "Assets", "value"].item() == 980.0


def test_growth_metrics_use_normalized_discrete_quarters() -> None:
    growth = Growth(ticker="AAPL", fundamentals=_sample_statements())
    expected = pd.Series(
        [140.0 / 110.0 - 1, 170.0 / 140.0 - 1, 230.0 / 170.0 - 1],
        index=pd.to_datetime(["2024-06-30", "2024-09-30", "2024-12-31"]),
        name="revenue_growth",
    )

    pd.testing.assert_series_equal(growth.revenue_growth(), expected)


def test_metric_as_of_prevents_future_filing_lookahead() -> None:
    growth = Growth(
        ticker="AAPL",
        fundamentals=_sample_statements(),
        as_of="2024-04-30",
    )

    with pytest.raises(ValueError, match="smaller than the number of observations"):
        growth.revenue_growth()


def test_period_selection_accepts_fundamentals_bundle() -> None:
    bundle = {
        "statements": _sample_statements(),
        "company_profile": pd.DataFrame(),
    }

    selected = select_fundamental_periods(bundle, frequency="annual")

    assert len(selected) == 2


def test_period_selection_rejects_unknown_frequency() -> None:
    with pytest.raises(ValueError, match="frequency"):
        select_fundamental_periods(
            _sample_statements(),
            frequency="monthly",  # type: ignore[arg-type]
        )


def test_period_helpers_accept_empty_fundamentals_bundle() -> None:
    bundle = {
        "statements": pd.DataFrame(),
        "company_profile": pd.DataFrame(),
    }

    assert select_fundamental_periods(bundle).empty
    assert build_ttm_fundamentals(bundle).empty
