import json
from pathlib import Path

import pandas as pd
import pytest

from fintern.data.exceptions import (
    MissingAPIKeyError,
    MissingDependencyError,
    NoProviderConfiguredError,
)
from fintern.data.providers import registry as registry_module
from fintern.data.providers.base import ProviderBase
from fintern.data.providers.openfigi import OpenFIGIProvider
from fintern.data.providers.sec import SECProvider
from fintern.data.providers.yahoo import _normalize_downloaded_market_data

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "data" / "providers"


def _load_json_fixture(*parts: str) -> object:
    fixture_path = _FIXTURE_ROOT.joinpath(*parts)
    return json.loads(fixture_path.read_text(encoding="utf-8"))


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


class _MissingDependencyMarketProvider(ProviderBase):
    name = "missing_dep"
    supports_market = True
    required_dependencies = ("package_that_will_not_exist_12345",)


class _ReadyMarketProvider(ProviderBase):
    name = "ready_market"
    supports_market = True


class _MissingKeyInstrumentProvider(ProviderBase):
    name = "needs_key"
    supports_instruments = True
    required_env_vars = ("FINTERN_TEST_PROVIDER_KEY",)


class _UnsupportedProvider(ProviderBase):
    name = "unsupported"


def test_registry_selects_first_available_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_module,
        "_PROVIDER_TYPES",
        (_MissingDependencyMarketProvider, _ReadyMarketProvider),
    )

    provider = registry_module.get_provider(None, "market")

    assert isinstance(provider, _ReadyMarketProvider)


def test_registry_requested_provider_missing_dependency_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_module,
        "_PROVIDER_TYPES",
        (_MissingDependencyMarketProvider,),
    )

    with pytest.raises(MissingDependencyError, match="fintern\\[missing_dep\\]"):
        registry_module.get_provider("missing_dep", "market")


def test_registry_requested_provider_missing_api_key_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_module,
        "_PROVIDER_TYPES",
        (_MissingKeyInstrumentProvider,),
    )
    monkeypatch.delenv("FINTERN_TEST_PROVIDER_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="FINTERN_TEST_PROVIDER_KEY"):
        registry_module.get_provider("needs_key", "instruments")


def test_registry_no_provider_available_raises_combined_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_module,
        "_PROVIDER_TYPES",
        (
            _UnsupportedProvider,
            _MissingDependencyMarketProvider,
            _MissingKeyInstrumentProvider,
        ),
    )
    monkeypatch.delenv("FINTERN_TEST_PROVIDER_KEY", raising=False)

    with pytest.raises(NoProviderConfiguredError) as exc_info:
        registry_module.get_provider(None, "market")

    message = str(exc_info.value)
    assert "unsupported" in message
    assert "missing_dep" in message
    assert "needs_key" in message


def test_yahoo_provider_normalizes_multiindex_download_frame() -> None:
    normalized = _normalize_downloaded_market_data(
        _mock_yfinance_download_frame(),
        ["AAPL", "MSFT"],
    )
    expected = pd.read_csv(
        _FIXTURE_ROOT / "yahoo" / "market.csv",
        parse_dates=["date"],
    )

    pd.testing.assert_frame_equal(normalized, expected)


def test_sec_provider_normalizes_fixture_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    company_tickers = _load_json_fixture("sec", "company_tickers.json")
    company_facts = _load_json_fixture("sec", "companyfacts_aapl.json")

    monkeypatch.setattr(SECProvider, "required_dependencies", ())
    provider = SECProvider(session=object())

    def _fake_get_json(url: str) -> dict[str, object]:
        if url == SECProvider.company_tickers_url:
            return company_tickers

        if url == SECProvider.company_facts_url.format(cik="0000320193"):
            return company_facts

        raise AssertionError(f"Unexpected SEC URL: {url}")

    monkeypatch.setattr(provider, "_get_json", _fake_get_json)

    downloaded = provider.download_fundamentals(["AAPL"])
    statements = downloaded["statements"]
    company_profile = downloaded["company_profile"]

    assert set(statements["statement"]) == {
        "income_statement",
        "balance_sheet",
        "cash_flow",
    }
    assert set(statements["metric"]) == {
        "Revenues",
        "Assets",
        "NetCashProvidedByUsedInOperatingActivities",
    }
    assert company_profile.loc[0, "company_name"] == "Apple Inc."


def test_openfigi_provider_normalizes_fixture_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping_payload = _load_json_fixture("openfigi", "mapping_response.json")

    monkeypatch.setattr(OpenFIGIProvider, "required_dependencies", ())
    monkeypatch.setattr(OpenFIGIProvider, "required_env_vars", ())
    provider = OpenFIGIProvider(session=object())
    monkeypatch.setattr(provider, "_post_json", lambda url, payload: mapping_payload)

    resolved = provider.resolve_instruments(["AAPL", "MSFT"])

    assert resolved["resolution_status"].tolist() == ["resolved", "resolved"]
    assert resolved["figi"].tolist() == ["BBG000B9XRY4", "BBG000BPH459"]
