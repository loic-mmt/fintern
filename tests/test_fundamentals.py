from pathlib import Path

import pandas as pd
import pytest

from fintern.data import fundamentals as fundamentals_module
from fintern.data.fundamentals import (
    FundamentalsData,
    build_company_dataset,
)


def _sample_fundamentals_bundle() -> dict[str, pd.DataFrame]:
    statements = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "statement": [
                "income_statement",
                "income_statement",
                "balance_sheet",
            ],
            "metric": ["Revenues", "NetIncomeLoss", "Assets"],
            "value": [1000.0, 250.0, 5000.0],
            "period_end": pd.to_datetime(
                ["2025-03-31", "2025-03-31", "2025-03-31"]
            ),
            "filed_date": pd.to_datetime(
                ["2025-05-01", "2025-05-01", "2025-05-01"]
            ),
            "fiscal_year": [2025, 2025, 2025],
            "fiscal_period": ["Q1", "Q1", "Q1"],
            "provider": ["sec", "sec", "sec"],
        }
    )
    company_profile = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "cik": ["0000320193"],
            "company_name": ["Apple Inc."],
            "provider": ["sec"],
        }
    )
    return {
        "statements": statements,
        "company_profile": company_profile,
    }


def _normalize_profile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy().reset_index(drop=True)

    for column_name in ("ticker", "cik", "company_name", "provider"):
        if column_name in normalized.columns:
            normalized[column_name] = normalized[column_name].astype(str)

    return normalized


@pytest.mark.parametrize("file_type", ["csv", "parquet"])
def test_download_fundamentals_saves_and_loads_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_type: str,
) -> None:
    if file_type == "parquet":
        pytest.importorskip("pyarrow")

    bundle = _sample_fundamentals_bundle()

    class _DummyProvider:
        def download_fundamentals(self, tickers, statements=None):
            del tickers, statements
            return {
                "statements": bundle["statements"].copy(),
                "company_profile": bundle["company_profile"].copy(),
            }

    monkeypatch.setattr(
        fundamentals_module,
        "get_provider",
        lambda provider, capability: _DummyProvider(),
    )

    output_path = tmp_path / f"fundamentals-{file_type}"
    downloaded = FundamentalsData.download_fundamentals(
        tickers="AAPL",
        path=output_path,
        file_type=file_type,
    )
    loaded = FundamentalsData.load_fundamentals(output_path)

    expected_statements = downloaded["statements"].sort_values(
        ["ticker", "statement", "metric"]
    ).reset_index(drop=True)
    loaded_statements = loaded["statements"].drop(columns=["source_path"]).sort_values(
        ["ticker", "statement", "metric"]
    ).reset_index(drop=True)
    for column_name in ("period_end", "filed_date"):
        expected_statements[column_name] = pd.to_datetime(
            expected_statements[column_name]
        )
        loaded_statements[column_name] = pd.to_datetime(loaded_statements[column_name])
    expected_statements["fiscal_year"] = expected_statements["fiscal_year"].astype(int)
    loaded_statements["fiscal_year"] = loaded_statements["fiscal_year"].astype(int)

    pd.testing.assert_frame_equal(loaded_statements, expected_statements)
    pd.testing.assert_frame_equal(
        _normalize_profile_frame(loaded["company_profile"]),
        _normalize_profile_frame(downloaded["company_profile"]),
    )


def test_build_company_dataset_asof_join_uses_filed_date() -> None:
    market_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-15", "2025-05-02"]),
            "ticker": ["AAPL", "AAPL"],
            "close": [100.0, 105.0],
        }
    )
    fundamentals = _sample_fundamentals_bundle()

    combined = build_company_dataset(market_data, fundamentals, join="asof")

    assert pd.isna(combined.loc[0, "income_statement__Revenues"])
    assert combined.loc[1, "income_statement__Revenues"] == 1000.0
    assert combined["profile__company_name"].tolist() == ["Apple Inc.", "Apple Inc."]


def test_build_company_dataset_period_end_join_uses_period_end_date() -> None:
    market_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-15", "2025-05-02"]),
            "ticker": ["AAPL", "AAPL"],
            "close": [100.0, 105.0],
        }
    )
    fundamentals = _sample_fundamentals_bundle()

    combined = build_company_dataset(market_data, fundamentals, join="period_end")

    assert combined.loc[0, "income_statement__Revenues"] == 1000.0
    assert combined.loc[1, "balance_sheet__Assets"] == 5000.0
