from __future__ import annotations

from pathlib import Path

from fintern.data.fundamentals import download_fundamentals
from fintern.data.instruments import resolve_instruments
from fintern.data.market import download_market_data


def main() -> None:
    fixture_root = Path(__file__).resolve().parent
    market_path = fixture_root / "market_refresh"
    fundamentals_path = fixture_root / "fundamentals_refresh"
    instruments_path = fixture_root / "instruments_refresh.csv"

    download_market_data(
        "AAPL",
        start="2025-01-01",
        end="2025-01-10",
        path=market_path,
        file_type="csv",
        provider="yahoo",
    )
    download_fundamentals(
        "AAPL",
        path=fundamentals_path,
        file_type="csv",
        provider="sec",
    )
    resolve_instruments(
        "AAPL",
        path=instruments_path,
        file_type="csv",
        provider="openfigi",
    )


if __name__ == "__main__":
    main()
