import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from fintern.data import market as market_module
from fintern.data.market import MarketData
from fintern.data.providers.yahoo import _normalize_downloaded_market_data


def _mock_yfinance_download_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = pd.MultiIndex.from_product(
        [
            ["AAPL", "MSFT"],
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        ]
    )
    rows = [
        [
            100.0,
            101.0,
            99.0,
            100.5,
            100.4,
            1000,
            200.0,
            201.0,
            199.0,
            200.5,
            200.4,
            2000,
        ],
        [
            101.0,
            102.0,
            100.0,
            101.5,
            101.4,
            1100,
            201.0,
            202.0,
            200.0,
            201.5,
            201.4,
            2100,
        ],
    ]

    return pd.DataFrame(rows, index=dates, columns=columns)


def test_load_market_data_reads_single_csv_file(tmp_path: Path) -> None:
    frame = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "close": [100.0, 101.5]})
    csv_path = tmp_path / "market.csv"
    frame.to_csv(csv_path, index=False)

    loaded = MarketData.load_market_data(csv_path)

    pd.testing.assert_frame_equal(loaded, frame)


def test_load_market_data_reads_nested_csv_files(tmp_path: Path) -> None:
    root_path = tmp_path / "market-data"
    first_folder = root_path / "AAPL"
    second_folder = root_path / "MSFT"
    first_folder.mkdir(parents=True)
    second_folder.mkdir(parents=True)

    first_frame = pd.DataFrame({"date": ["2025-01-01"], "close": [100.0]})
    second_frame = pd.DataFrame({"date": ["2025-01-02"], "close": [101.0]})

    first_frame.to_csv(first_folder / "2025.csv", index=False)
    second_frame.to_csv(second_folder / "2025.csv", index=False)

    loaded = MarketData.load_market_data(root_path)

    expected = pd.DataFrame(
        {
            "source_path": ["AAPL/2025.csv", "MSFT/2025.csv"],
            "date": ["2025-01-01", "2025-01-02"],
            "close": [100.0, 101.0],
        }
    )

    pd.testing.assert_frame_equal(
        loaded.sort_values("source_path").reset_index(drop=True),
        expected.sort_values("source_path").reset_index(drop=True),
    )


def test_load_market_data_reads_partitioned_parquet_folder(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    root_path = tmp_path / "parquet-data"

    first_year = root_path / "year=2024"
    second_year = root_path / "year=2025"
    first_year.mkdir(parents=True)
    second_year.mkdir(parents=True)

    pd.DataFrame({"ticker": ["AAPL"], "close": [100.0]}).to_parquet(
        first_year / "part-000.parquet",
        index=False,
    )
    pd.DataFrame({"ticker": ["MSFT"], "close": [101.0]}).to_parquet(
        second_year / "part-001.parquet",
        index=False,
    )

    loaded = (
        MarketData.load_market_data(root_path)
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    expected = pd.DataFrame(
        {"ticker": ["AAPL", "MSFT"], "close": [100.0, 101.0], "year": [2024, 2025]}
    )

    loaded["year"] = loaded["year"].astype(int)

    pd.testing.assert_frame_equal(loaded, expected)


def test_download_market_data_saves_partitioned_csv_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded_frame = _normalize_downloaded_market_data(
        _mock_yfinance_download_frame(),
        ["AAPL", "MSFT"],
    )

    class _DummyProvider:
        def download_market_data(self, tickers, start=None, end=None, interval="1d"):
            del tickers, start, end, interval
            return downloaded_frame.copy()

    monkeypatch.setattr(
        market_module,
        "get_provider",
        lambda provider, capability: _DummyProvider(),
    )

    output_path = tmp_path / "csv-data"
    downloaded = MarketData.download_market_data(
        tickers=["aapl", "msft"],
        start="2025-01-01",
        end="2025-01-10",
        path=output_path,
        file_type="csv",
    )

    assert set(downloaded["ticker"]) == {"AAPL", "MSFT"}
    assert (output_path / "ticker=AAPL" / "year=2025" / "data.csv").exists()
    assert (output_path / "ticker=MSFT" / "year=2025" / "data.csv").exists()

    loaded = MarketData.load_market_data(output_path).sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    expected = downloaded.assign(year=downloaded["date"].dt.year)[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "year",
        ]
    ].sort_values(["ticker", "date"]).reset_index(drop=True)

    loaded = loaded.drop(columns=["source_path"])[expected.columns]
    loaded["date"] = pd.to_datetime(loaded["date"])
    loaded["year"] = loaded["year"].astype(int)
    expected["year"] = expected["year"].astype(int)

    pd.testing.assert_frame_equal(loaded, expected)


def test_download_market_data_saves_partitioned_parquet_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")

    downloaded_frame = _normalize_downloaded_market_data(
        _mock_yfinance_download_frame(),
        ["AAPL", "MSFT"],
    )

    class _DummyProvider:
        def download_market_data(self, tickers, start=None, end=None, interval="1d"):
            del tickers, start, end, interval
            return downloaded_frame.copy()

    monkeypatch.setattr(
        market_module,
        "get_provider",
        lambda provider, capability: _DummyProvider(),
    )

    output_path = tmp_path / "parquet-data"
    downloaded = MarketData.download_market_data(
        tickers=["AAPL", "MSFT"],
        start="2025-01-01",
        end="2025-01-10",
        path=output_path,
        file_type="parquet",
    )

    loaded = MarketData.load_market_data(output_path).sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)
    expected = downloaded.assign(year=downloaded["date"].dt.year).sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    loaded = loaded[expected.columns]
    loaded["ticker"] = loaded["ticker"].astype(str)
    expected["ticker"] = expected["ticker"].astype(str)
    loaded["year"] = loaded["year"].astype(int)
    expected["year"] = expected["year"].astype(int)

    pd.testing.assert_frame_equal(loaded, expected)


def test_market_data_is_not_dataclass() -> None:
    assert not dataclasses.is_dataclass(MarketData)
