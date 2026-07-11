from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from fintern.data.providers.registry import get_provider
from fintern.data.storage import load_tabular_dataset, save_to_csv, save_to_parquet


def _normalize_tickers(tickers: str | Sequence[str]) -> list[str]:
    if isinstance(tickers, str):
        raw_tickers = tickers.replace(",", " ").split()
    else:
        raw_tickers = [str(ticker) for ticker in tickers]

    normalized_tickers = [
        ticker.strip().upper() for ticker in raw_tickers if ticker.strip()
    ]

    if not normalized_tickers:
        raise ValueError("tickers cannot be empty")

    return list(dict.fromkeys(normalized_tickers))


def _save_market_data(
    data: pd.DataFrame,
    path: str | Path,
    file_type: Literal["csv", "parquet"],
) -> Path:
    if file_type == "csv":
        return save_to_csv(data=data, path=path, partition_cols=("ticker", "year"))

    return save_to_parquet(data=data, path=path, partition_cols=("ticker", "year"))


class MarketData:
    """Load or download normalized market data."""

    @staticmethod
    def load_market_data(path: str | Path) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return load_tabular_dataset(path)

    @staticmethod
    def download_market_data(
        tickers: str | Sequence[str],
        start: str | None = None,
        end: str | None = None,
        path: str | Path | None = None,
        file_type: Literal["csv", "parquet"] = "parquet",
        interval: str = "1d",
        provider: str | None = None,
    ) -> pd.DataFrame:
        normalized_tickers = _normalize_tickers(tickers)
        provider_client = get_provider(provider=provider, capability="market")
        data = provider_client.download_market_data(
            tickers=normalized_tickers,
            start=start,
            end=end,
            interval=interval,
        )

        if path is None:
            return data

        _save_market_data(data=data, path=path, file_type=file_type)
        return data


def load_market_data(path: str | Path) -> pd.DataFrame | dict[str, pd.DataFrame]:
    return MarketData.load_market_data(path)


def download_market_data(
    tickers: str | Sequence[str],
    start: str | None = None,
    end: str | None = None,
    path: str | Path | None = None,
    file_type: Literal["csv", "parquet"] = "parquet",
    interval: str = "1d",
    provider: str | None = None,
) -> pd.DataFrame:
    return MarketData.download_market_data(
        tickers=tickers,
        start=start,
        end=end,
        path=path,
        file_type=file_type,
        interval=interval,
        provider=provider,
    )


__all__ = ["MarketData", "download_market_data", "load_market_data"]
