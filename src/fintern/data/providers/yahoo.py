from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from fintern.data.providers.base import ProviderBase

_YFINANCE_PRICE_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


def _normalize_column_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _normalize_downloaded_market_data(
    raw_data: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    if raw_data.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )

    normalized = raw_data.copy()
    normalized.index = pd.to_datetime(normalized.index)
    index_name = normalized.index.name or "date"

    if isinstance(normalized.columns, pd.MultiIndex):
        first_level = {str(value) for value in normalized.columns.get_level_values(0)}
        second_level = {str(value) for value in normalized.columns.get_level_values(1)}

        if _YFINANCE_PRICE_FIELDS & first_level:
            normalized = normalized.stack(level=1, future_stack=True)
        elif _YFINANCE_PRICE_FIELDS & second_level:
            normalized = normalized.stack(level=0, future_stack=True)
        else:
            raise ValueError("Unexpected yfinance download format.")

        normalized = normalized.rename_axis(index=[index_name, "ticker"]).reset_index()
    else:
        normalized = normalized.reset_index()
        normalized["ticker"] = tickers[0]

    normalized = normalized.rename(columns={index_name: "date"})
    normalized = normalized.rename(columns=_normalize_column_name)
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()

    ordered_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    present_ordered_columns = [
        column_name
        for column_name in ordered_columns
        if column_name in normalized.columns
    ]
    remaining_columns = [
        column_name
        for column_name in normalized.columns
        if column_name not in present_ordered_columns
    ]

    return normalized[present_ordered_columns + remaining_columns].sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


class YahooProvider(ProviderBase):
    name = "yahoo"
    supports_market = True
    required_dependencies = ("yfinance",)

    def download_market_data(
        self,
        tickers: Sequence[str],
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.ensure_available("market")

        import yfinance as yf

        raw_data = yf.download(
            tickers=list(tickers),
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            progress=False,
        )
        return _normalize_downloaded_market_data(raw_data, tickers)
