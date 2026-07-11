from __future__ import annotations

from fintern.data.providers.base import ProviderBase


class EODHDProvider(ProviderBase):
    name = "eodhd"
    required_dependencies = ("requests",)
    required_env_vars = ("FINTERN_EODHD_API_KEY",)
    notes = ("Adapter scaffolded for fintern v1; not implemented yet.",)
