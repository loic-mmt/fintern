from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from fintern.data.providers.registry import get_provider
from fintern.data.storage import load_tabular_dataset, save_to_csv, save_to_parquet


def _normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
    if isinstance(symbols, str):
        raw_symbols = symbols.replace(",", " ").split()
    else:
        raw_symbols = [str(symbol) for symbol in symbols]

    normalized_symbols = [
        symbol.strip().upper() for symbol in raw_symbols if symbol.strip()
    ]

    if not normalized_symbols:
        raise ValueError("symbols cannot be empty")

    return list(dict.fromkeys(normalized_symbols))


def _resolve_flat_table_path(path: Path) -> Path:
    if not path.is_dir():
        return path

    for file_name in ("data.parquet", "data.csv", "data.csv.gz"):
        candidate = path / file_name

        if candidate.exists():
            return candidate

    return path


def load_instruments(path: str | Path) -> pd.DataFrame | dict[str, pd.DataFrame]:
    return load_tabular_dataset(_resolve_flat_table_path(Path(path).expanduser()))


def resolve_instruments(
    symbols: str | Sequence[str],
    path: str | Path | None = None,
    file_type: Literal["csv", "parquet"] = "parquet",
    provider: str | None = None,
    exchange_code: str | None = None,
) -> pd.DataFrame:
    normalized_symbols = _normalize_symbols(symbols)
    provider_client = get_provider(provider=provider, capability="instruments")
    data = provider_client.resolve_instruments(
        symbols=normalized_symbols,
        exchange_code=exchange_code,
    )

    if path is None:
        return data

    if file_type == "csv":
        save_to_csv(data, path)
        return data

    save_to_parquet(data, path)
    return data


__all__ = ["load_instruments", "resolve_instruments"]
