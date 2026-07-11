from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import pandas as pd

from fintern.data.exceptions import InstrumentResolutionError
from fintern.data.providers.base import ProviderBase


class OpenFIGIProvider(ProviderBase):
    name = "openfigi"
    supports_instruments = True
    required_dependencies = ("requests",)
    required_env_vars = ("FINTERN_OPENFIGI_API_KEY",)
    mapping_url = "https://api.openfigi.com/v3/mapping"

    def __init__(self, session: Any | None = None) -> None:
        self._session = session or self._build_session()

    def _build_session(self) -> Any:
        import requests

        session = requests.Session()
        session.headers.update(
            {
                "Content-Type": "application/json",
                "X-OPENFIGI-APIKEY": os.environ["FINTERN_OPENFIGI_API_KEY"],
                "User-Agent": os.getenv(
                    "FINTERN_OPENFIGI_USER_AGENT",
                    "fintern/0.1.0",
                ),
            }
        )
        return session

    def _post_json(
        self,
        url: str,
        payload: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        response = self._session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list):
            raise InstrumentResolutionError("Unexpected OpenFIGI response format.")

        return data

    @staticmethod
    def _normalize_mapping_response(
        payload: list[dict[str, Any]],
        symbols: Sequence[str],
    ) -> pd.DataFrame:
        if len(payload) != len(symbols):
            raise InstrumentResolutionError(
                "OpenFIGI response count did not match the requested symbols."
            )

        rows: list[dict[str, Any]] = []

        for symbol, item in zip(symbols, payload, strict=True):
            match = (item.get("data") or [{}])[0]
            rows.append(
                {
                    "symbol": symbol,
                    "ticker": match.get("ticker"),
                    "name": match.get("name"),
                    "exchange": match.get("exchCode"),
                    "currency": match.get("currency"),
                    "figi": match.get("figi"),
                    "composite_figi": match.get("compositeFIGI"),
                    "share_class_figi": match.get("shareClassFIGI"),
                    "security_type": match.get("securityType"),
                    "market_sector": match.get("marketSector"),
                    "provider": "openfigi",
                    "resolution_status": "resolved" if match else "unresolved",
                    "error": item.get("error"),
                }
            )

        return pd.DataFrame(rows)

    def resolve_instruments(
        self,
        symbols: Sequence[str],
        exchange_code: str | None = None,
    ) -> pd.DataFrame:
        self.ensure_available("instruments")
        payload = []

        for symbol in symbols:
            job = {"idType": "TICKER", "idValue": symbol}

            if exchange_code:
                job["exchCode"] = exchange_code

            payload.append(job)

        response = self._post_json(self.mapping_url, payload)
        return self._normalize_mapping_response(response, symbols)
