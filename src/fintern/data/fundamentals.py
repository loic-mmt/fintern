from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from fintern.data.models import JoinMode, NormalizedFundamentals
from fintern.data.providers.registry import get_provider
from fintern.data.storage import load_tabular_dataset, save_to_csv, save_to_parquet

_STATEMENT_COLUMN_ORDER = [
    "ticker",
    "statement",
    "metric",
    "label",
    "description",
    "unit",
    "value",
    "period_start",
    "period_end",
    "filed_date",
    "fiscal_year",
    "fiscal_period",
    "form",
    "frame",
    "accession_number",
    "taxonomy",
    "provider",
    "period_type",
    "is_derived",
    "derivation",
]


def _normalize_tickers(tickers: str | Sequence[str]) -> list[str]:
    if isinstance(tickers, str):
        raw_tickers = tickers.replace(",", " ").split()
    else:
        raw_tickers = [str(ticker) for ticker in tickers]

    normalized_tickers = [
        ticker.strip().upper() for ticker in raw_tickers if ticker.strip()
    ]

    if not normalized_tickers:
        raise ValueError("tickers cannot be empty")

    return list(dict.fromkeys(normalized_tickers))


def _empty_fundamentals_bundle() -> NormalizedFundamentals:
    return {
        "statements": pd.DataFrame(),
        "company_profile": pd.DataFrame(),
    }


def _normalize_fundamentals_bundle(
    data: Mapping[str, pd.DataFrame],
) -> NormalizedFundamentals:
    statements = data.get("statements", pd.DataFrame())
    company_profile = data.get("company_profile", pd.DataFrame())

    if not isinstance(statements, pd.DataFrame):
        raise TypeError("`statements` fundamentals payload must be a DataFrame.")

    if not isinstance(company_profile, pd.DataFrame):
        raise TypeError("`company_profile` fundamentals payload must be a DataFrame.")

    return {
        "statements": statements.copy(),
        "company_profile": company_profile.copy(),
    }


def _normalize_output_root(path: str | Path) -> Path:
    output_path = Path(path).expanduser()

    if output_path.suffixes:
        raise ValueError(
            "Fundamentals datasets contain multiple tables and must be saved "
            "to a directory."
        )

    return output_path


def _prepare_statements_for_save(
    statements: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    prepared = statements.copy()

    if prepared.empty:
        return prepared, ("ticker", "statement")

    if "ticker" not in prepared.columns or "statement" not in prepared.columns:
        raise ValueError(
            "Fundamentals statements must include `ticker` and `statement` columns."
        )

    if "fiscal_year" not in prepared.columns and "period_end" in prepared.columns:
        prepared["fiscal_year"] = pd.to_datetime(prepared["period_end"]).dt.year.astype(
            "Int64"
        )

    if "fiscal_year" in prepared.columns:
        return prepared, ("ticker", "statement", "fiscal_year")

    if "fiscal_period" in prepared.columns:
        return prepared, ("ticker", "statement", "fiscal_period")

    return prepared, ("ticker", "statement")


def _save_fundamentals_bundle(
    data: Mapping[str, pd.DataFrame],
    path: str | Path,
    file_type: Literal["csv", "parquet"],
) -> Path:
    output_root = _normalize_output_root(path)
    bundle = _normalize_fundamentals_bundle(data)
    output_root.mkdir(parents=True, exist_ok=True)

    statements, partition_cols = _prepare_statements_for_save(bundle["statements"])
    if file_type == "csv":
        save_to_csv(
            statements,
            output_root / "statements",
            partition_cols=partition_cols,
        )
        save_to_csv(bundle["company_profile"], output_root / "company_profile.csv")
        return output_root

    save_to_parquet(
        statements,
        output_root / "statements",
        partition_cols=partition_cols,
    )
    save_to_parquet(bundle["company_profile"], output_root / "company_profile.parquet")
    return output_root


def _load_profile_table(root_path: Path) -> pd.DataFrame:
    for file_name in (
        "company_profile.parquet",
        "company_profile.csv",
        "company_profile.csv.gz",
    ):
        profile_path = root_path / file_name

        if profile_path.exists():
            loaded = load_tabular_dataset(profile_path)
            if isinstance(loaded, dict):
                raise ValueError("Company profile payload must load as a single table.")
            return _normalize_company_profile(loaded)

    return pd.DataFrame()


def _order_statement_columns(statements: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        column_name
        for column_name in _STATEMENT_COLUMN_ORDER
        if column_name in statements.columns
    ]
    remaining_columns = [
        column_name
        for column_name in statements.columns
        if column_name not in preferred_columns
    ]
    return statements[preferred_columns + remaining_columns]


def _normalize_company_profile(company_profile: pd.DataFrame) -> pd.DataFrame:
    normalized = company_profile.copy()

    if "ticker" in normalized.columns:
        normalized["ticker"] = normalized["ticker"].astype("string").str.upper()

    if "cik" in normalized.columns:
        cik = normalized["cik"].astype("string").str.replace(r"\.0$", "", regex=True)
        normalized["cik"] = cik.where(cik.isna(), cik.str.zfill(10))

    return normalized


class FundamentalsData:
    """Load or download normalized fundamentals data."""

    @staticmethod
    def load_fundamentals(path: str | Path) -> NormalizedFundamentals:
        normalized_path = Path(path).expanduser()

        if normalized_path.is_file():
            loaded = load_tabular_dataset(normalized_path)

            if isinstance(loaded, dict):
                raise ValueError(
                    "Fundamentals file payload must load as a single table."
                )

            return {"statements": loaded, "company_profile": pd.DataFrame()}

        statements_path = normalized_path / "statements"

        if statements_path.exists():
            statements = load_tabular_dataset(statements_path)

            if isinstance(statements, dict):
                raise ValueError(
                    "Fundamentals statements dataset must load as a single DataFrame."
                )

            return {
                "statements": _order_statement_columns(statements),
                "company_profile": _load_profile_table(normalized_path),
            }

        loaded = load_tabular_dataset(normalized_path)

        if isinstance(loaded, dict):
            raise ValueError(
                "Unrecognized fundamentals folder layout. Expected a "
                "`statements` dataset."
            )

        return {
            "statements": _order_statement_columns(loaded),
            "company_profile": pd.DataFrame(),
        }

    @staticmethod
    def download_fundamentals(
        tickers: str | Sequence[str],
        path: str | Path | None = None,
        file_type: Literal["csv", "parquet"] = "parquet",
        provider: str | None = None,
        statements: Sequence[str] | None = None,
    ) -> NormalizedFundamentals:
        normalized_tickers = _normalize_tickers(tickers)
        provider_client = get_provider(provider=provider, capability="fundamentals")
        data = provider_client.download_fundamentals(
            tickers=normalized_tickers,
            statements=statements,
        )
        normalized = _normalize_fundamentals_bundle(data)

        if path is None:
            return normalized

        _save_fundamentals_bundle(normalized, path=path, file_type=file_type)
        return normalized


def _coerce_statements_frame(
    fundamentals: pd.DataFrame | Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if isinstance(fundamentals, pd.DataFrame):
        return fundamentals.copy()

    bundle = _normalize_fundamentals_bundle(fundamentals)
    return bundle["statements"]


def _coerce_profile_frame(
    fundamentals: pd.DataFrame | Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if isinstance(fundamentals, pd.DataFrame):
        return pd.DataFrame()

    bundle = _normalize_fundamentals_bundle(fundamentals)
    return bundle["company_profile"]


def build_company_dataset(
    market_data: pd.DataFrame,
    fundamentals: pd.DataFrame | Mapping[str, pd.DataFrame],
    join: JoinMode = "asof",
) -> pd.DataFrame:
    if join not in {"asof", "period_end"}:
        raise ValueError("`join` must be either `asof` or `period_end`.")

    if "date" not in market_data.columns or "ticker" not in market_data.columns:
        raise ValueError("`market_data` must include `date` and `ticker` columns.")

    statements = _coerce_statements_frame(fundamentals)
    company_profile = _coerce_profile_frame(fundamentals)
    combined = market_data.copy()
    combined["date"] = pd.to_datetime(combined["date"])

    if statements.empty:
        result = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    else:
        required_columns = {"ticker", "statement", "metric", "value", "period_end"}
        missing_columns = required_columns - set(statements.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Fundamentals statements are missing required columns: {missing}"
            )

        join_key = (
            "filed_date"
            if join == "asof" and "filed_date" in statements.columns
            else "period_end"
        )
        normalized = statements.copy()
        normalized[join_key] = pd.to_datetime(normalized[join_key])
        normalized["feature_name"] = (
            normalized["statement"].astype(str)
            + "__"
            + normalized["metric"].astype(str)
        )
        wide = (
            normalized.pivot_table(
                index=["ticker", join_key],
                columns="feature_name",
                values="value",
                aggfunc="last",
            )
            .reset_index()
            .sort_values(["ticker", join_key])
        )
        wide.columns.name = None

        result = pd.merge_asof(
            combined.sort_values(["ticker", "date"]),
            wide,
            left_on="date",
            right_on=join_key,
            by="ticker",
            direction="backward",
        ).reset_index(drop=True)

    if not company_profile.empty and "ticker" in company_profile.columns:
        profile = company_profile.drop_duplicates(subset=["ticker"]).copy()
        renamed_columns = {
            column_name: f"profile__{column_name}"
            for column_name in profile.columns
            if column_name != "ticker"
        }
        profile = profile.rename(columns=renamed_columns)
        result = result.merge(profile, on="ticker", how="left")

    return result


def load_fundamentals(path: str | Path) -> NormalizedFundamentals:
    return FundamentalsData.load_fundamentals(path)


def download_fundamentals(
    tickers: str | Sequence[str],
    path: str | Path | None = None,
    file_type: Literal["csv", "parquet"] = "parquet",
    provider: str | None = None,
    statements: Sequence[str] | None = None,
) -> NormalizedFundamentals:
    return FundamentalsData.download_fundamentals(
        tickers=tickers,
        path=path,
        file_type=file_type,
        provider=provider,
        statements=statements,
    )


__all__ = [
    "FundamentalsData",
    "build_company_dataset",
    "download_fundamentals",
    "load_fundamentals",
]
