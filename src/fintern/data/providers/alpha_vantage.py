from __future__ import annotations

from fintern.data.providers.base import ProviderBase


class AlphaVantageProvider(ProviderBase):
    name = "alpha_vantage"
    required_dependencies = ("requests",)
    required_env_vars = ("FINTERN_ALPHA_VANTAGE_API_KEY",)
    notes = ("Adapter scaffolded for fintern v1; not implemented yet.",)
