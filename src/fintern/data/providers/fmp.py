from __future__ import annotations

from fintern.data.providers.base import ProviderBase


class FMPProvider(ProviderBase):
    name = "fmp"
    required_dependencies = ("requests",)
    required_env_vars = ("FINTERN_FMP_API_KEY",)
    notes = ("Adapter scaffolded for fintern v1; not implemented yet.",)
