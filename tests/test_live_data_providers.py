import os

import pytest

from fintern.data.fundamentals import download_fundamentals
from fintern.data.instruments import resolve_instruments
from fintern.data.market import download_market_data

pytestmark = pytest.mark.live_api


def _live_api_enabled() -> bool:
    return os.getenv("FINTERN_RUN_LIVE_API_TESTS") == "1"


def test_live_yahoo_market_download() -> None:
    if not _live_api_enabled():
        pytest.skip("Set FINTERN_RUN_LIVE_API_TESTS=1 to run live provider tests.")

    pytest.importorskip("yfinance")
    data = download_market_data(
        "AAPL",
        start="2025-01-01",
        end="2025-01-10",
        provider="yahoo",
    )
    assert not data.empty


def test_live_sec_fundamentals_download() -> None:
    if not _live_api_enabled():
        pytest.skip("Set FINTERN_RUN_LIVE_API_TESTS=1 to run live provider tests.")

    data = download_fundamentals("AAPL", provider="sec")
    assert not data["statements"].empty


def test_live_openfigi_resolution() -> None:
    if not _live_api_enabled():
        pytest.skip("Set FINTERN_RUN_LIVE_API_TESTS=1 to run live provider tests.")

    if not os.getenv("FINTERN_OPENFIGI_API_KEY"):
        pytest.skip("Set FINTERN_OPENFIGI_API_KEY to run live OpenFIGI tests.")

    resolved = resolve_instruments("AAPL", provider="openfigi")
    assert not resolved.empty
