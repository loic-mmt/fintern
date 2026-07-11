from __future__ import annotations

from fintern.data.exceptions import NoProviderConfiguredError, ProviderNotAvailableError
from fintern.data.models import Capability, ProviderAvailability
from fintern.data.providers.alpha_vantage import AlphaVantageProvider
from fintern.data.providers.base import ProviderBase
from fintern.data.providers.eodhd import EODHDProvider
from fintern.data.providers.fmp import FMPProvider
from fintern.data.providers.openfigi import OpenFIGIProvider
from fintern.data.providers.sec import SECProvider
from fintern.data.providers.yahoo import YahooProvider

_PROVIDER_TYPES: tuple[type[ProviderBase], ...] = (
    YahooProvider,
    SECProvider,
    OpenFIGIProvider,
    FMPProvider,
    EODHDProvider,
    AlphaVantageProvider,
)


def _normalize_provider_name(provider: str) -> str:
    return provider.strip().lower().replace("-", "_").replace(" ", "_")


def _provider_map() -> dict[str, type[ProviderBase]]:
    return {
        _normalize_provider_name(provider.name): provider
        for provider in _PROVIDER_TYPES
    }


def list_provider_availability() -> list[ProviderAvailability]:
    return [provider.availability() for provider in _PROVIDER_TYPES]


def _format_provider_issue(
    availability: ProviderAvailability,
    capability: Capability,
) -> str:
    issues: list[str] = []

    if not availability.supports(capability):
        issues.append("unsupported capability")

    if availability.missing_dependencies:
        dependencies = ", ".join(availability.missing_dependencies)
        issues.append(f"missing dependencies: {dependencies}")

    if availability.missing_api_keys:
        env_vars = ", ".join(availability.missing_api_keys)
        issues.append(f"missing configuration: {env_vars}")

    issues.extend(availability.notes)
    return "; ".join(issues) if issues else "available"


def _raise_no_provider_available(capability: Capability) -> None:
    lines = [f"No available provider could satisfy `{capability}`."]

    for provider in _PROVIDER_TYPES:
        availability = provider.availability()
        lines.append(
            f"- {availability.name}: {_format_provider_issue(availability, capability)}"
        )

    raise NoProviderConfiguredError("\n".join(lines))


def get_provider(
    provider: str | None,
    capability: Capability,
) -> ProviderBase:
    if provider is not None:
        provider_type = _provider_map().get(_normalize_provider_name(provider))

        if provider_type is None:
            known_providers = ", ".join(sorted(_provider_map()))
            raise ProviderNotAvailableError(
                f"Unknown provider `{provider}`. Known providers: {known_providers}."
            )

        provider_type.ensure_available(capability)
        return provider_type()

    for provider_type in _PROVIDER_TYPES:
        availability = provider_type.availability()

        if availability.ready_for(capability):
            return provider_type()

    _raise_no_provider_available(capability)
