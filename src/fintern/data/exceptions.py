from __future__ import annotations

from collections.abc import Sequence


class DataLayerError(Exception):
    """Base error for fintern's data layer."""


class ProviderNotAvailableError(DataLayerError):
    """Raised when a requested provider cannot be used."""


class MissingDependencyError(ProviderNotAvailableError):
    """Raised when a provider dependency is not installed."""

    def __init__(self, provider: str, dependencies: Sequence[str]) -> None:
        missing = ", ".join(dependencies)
        super().__init__(
            f"Provider `{provider}` is unavailable because dependencies are "
            f"missing: {missing}. Install them with `pip install fintern[{provider}]`."
        )


class MissingAPIKeyError(ProviderNotAvailableError):
    """Raised when a provider requires missing environment variables."""

    def __init__(self, provider: str, env_vars: Sequence[str]) -> None:
        missing = ", ".join(env_vars)
        super().__init__(
            f"Provider `{provider}` is unavailable because configuration is "
            f"missing. Set these environment variables: {missing}."
        )


class NoProviderConfiguredError(ProviderNotAvailableError):
    """Raised when no provider can satisfy the requested capability."""


class UnsupportedCapabilityError(DataLayerError):
    """Raised when a provider does not implement a capability."""


class InstrumentResolutionError(DataLayerError):
    """Raised when a symbol cannot be resolved to an instrument."""
