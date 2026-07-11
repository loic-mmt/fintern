from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd

Capability = Literal["market", "fundamentals", "instruments"]
JoinMode = Literal["asof", "period_end"]
NormalizedFundamentals: TypeAlias = dict[str, pd.DataFrame]


@dataclass(frozen=True)
class ProviderAvailability:
    name: str
    supports_market: bool = False
    supports_fundamentals: bool = False
    supports_instruments: bool = False
    missing_dependencies: tuple[str, ...] = ()
    missing_api_keys: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def supports(self, capability: Capability) -> bool:
        if capability == "market":
            return self.supports_market

        if capability == "fundamentals":
            return self.supports_fundamentals

        return self.supports_instruments

    def ready_for(self, capability: Capability) -> bool:
        return (
            self.supports(capability)
            and not self.missing_dependencies
            and not self.missing_api_keys
        )
