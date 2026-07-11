from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence

import pandas as pd

from fintern.data.exceptions import (
    MissingAPIKeyError,
    MissingDependencyError,
    UnsupportedCapabilityError,
)
from fintern.data.models import Capability, NormalizedFundamentals, ProviderAvailability


class ProviderBase:
    name = "base"
    supports_market = False
    supports_fundamentals = False
    supports_instruments = False
    required_dependencies: tuple[str, ...] = ()
    required_env_vars: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @classmethod
    def supports_capability(cls, capability: Capability) -> bool:
        if capability == "market":
            return cls.supports_market

        if capability == "fundamentals":
            return cls.supports_fundamentals

        return cls.supports_instruments

    @classmethod
    def availability(cls) -> ProviderAvailability:
        missing_dependencies = tuple(
            dependency
            for dependency in cls.required_dependencies
            if importlib.util.find_spec(dependency) is None
        )
        missing_api_keys = tuple(
            env_var
            for env_var in cls.required_env_vars
            if not os.getenv(env_var)
        )

        return ProviderAvailability(
            name=cls.name,
            supports_market=cls.supports_market,
            supports_fundamentals=cls.supports_fundamentals,
            supports_instruments=cls.supports_instruments,
            missing_dependencies=missing_dependencies,
            missing_api_keys=missing_api_keys,
            notes=cls.notes,
        )

    @classmethod
    def ensure_available(cls, capability: Capability) -> None:
        availability = cls.availability()

        if not availability.supports(capability):
            raise UnsupportedCapabilityError(
                f"Provider `{cls.name}` does not support `{capability}`."
            )

        if availability.missing_dependencies:
            raise MissingDependencyError(cls.name, availability.missing_dependencies)

        if availability.missing_api_keys:
            raise MissingAPIKeyError(cls.name, availability.missing_api_keys)

    def download_market_data(
        self,
        tickers: Sequence[str],
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        del tickers, start, end, interval
        self.ensure_available("market")
        raise UnsupportedCapabilityError(
            f"Provider `{self.name}` does not implement market downloads."
        )

    def download_fundamentals(
        self,
        tickers: Sequence[str],
        statements: Sequence[str] | None = None,
    ) -> NormalizedFundamentals:
        del tickers, statements
        self.ensure_available("fundamentals")
        raise UnsupportedCapabilityError(
            f"Provider `{self.name}` does not implement fundamentals downloads."
        )

    def resolve_instruments(
        self,
        symbols: Sequence[str],
        exchange_code: str | None = None,
    ) -> pd.DataFrame:
        del symbols, exchange_code
        self.ensure_available("instruments")
        raise UnsupportedCapabilityError(
            f"Provider `{self.name}` does not implement instrument resolution."
        )
