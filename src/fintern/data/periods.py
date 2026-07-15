from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias

import pandas as pd

FiscalFrequency = Literal["all", "quarterly", "annual"]
StatementsInput: TypeAlias = pd.DataFrame | Mapping[str, pd.DataFrame]

_FLOW_STATEMENTS = {"income_statement", "cash_flow"}
_ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
_IDENTITY_COLUMNS = (
    "ticker",
    "statement",
    "metric",
    "unit",
    "provider",
    "taxonomy",
    "period_end",
    "period_type",
)
_METRIC_GROUP_COLUMNS = (
    "ticker",
    "statement",
    "metric",
    "unit",
    "provider",
    "taxonomy",
)


def _coerce_statements(data: StatementsInput) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        statements = data.copy()
    elif isinstance(data, Mapping):
        statements = data.get("statements", pd.DataFrame()).copy()
    else:
        raise TypeError("statements must be a DataFrame or a fundamentals mapping")

    if not isinstance(statements, pd.DataFrame):
        raise TypeError("fundamentals `statements` payload must be a DataFrame")

    required_columns = {"ticker", "statement", "metric", "value", "period_end"}
    missing_columns = sorted(required_columns - set(statements.columns))

    if statements.empty:
        for column_name in missing_columns:
            statements[column_name] = pd.Series(dtype="object")
        return statements

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"statements must contain columns: {missing}")

    return statements


def _normalize_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    return timestamp


def _normalize_statement_values(statements: pd.DataFrame) -> pd.DataFrame:
    normalized = statements.copy()
    normalized["ticker"] = normalized["ticker"].astype("string").str.upper()
    normalized["statement"] = normalized["statement"].astype("string").str.lower()
    normalized["metric"] = normalized["metric"].astype("string")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")

    for column_name in ("period_start", "period_end", "filed_date"):
        if column_name in normalized.columns:
            normalized[column_name] = pd.to_datetime(
                normalized[column_name],
                errors="coerce",
            ).dt.tz_localize(None)

    return normalized


def classify_fundamental_periods(statements: StatementsInput) -> pd.DataFrame:
    """Classify normalized statement rows by fiscal period semantics."""
    classified = _normalize_statement_values(_coerce_statements(statements))

    if classified.empty:
        classified["period_type"] = pd.Series(dtype="string")
        classified["is_derived"] = pd.Series(dtype="bool")
        classified["derivation"] = pd.Series(dtype="string")
        return classified

    period_type = pd.Series("unknown", index=classified.index, dtype="string")
    statement = classified["statement"]
    balance_sheet = statement.eq("balance_sheet")
    flow_statement = statement.isin(_FLOW_STATEMENTS)
    period_type.loc[balance_sheet] = "instant"

    if "period_start" in classified.columns:
        duration_days = (
            classified["period_end"] - classified["period_start"]
        ).dt.days.add(1)
        period_type.loc[flow_statement & duration_days.between(45, 135)] = "quarterly"
        period_type.loc[flow_statement & duration_days.between(136, 299)] = "ytd"
        period_type.loc[flow_statement & duration_days.between(300, 425)] = "annual"

    classified["period_type"] = period_type

    if "is_derived" not in classified.columns:
        classified["is_derived"] = False
    else:
        classified["is_derived"] = classified["is_derived"].fillna(False).astype(bool)

    if "derivation" not in classified.columns:
        classified["derivation"] = pd.Series(
            pd.NA,
            index=classified.index,
            dtype="string",
        )
    else:
        classified["derivation"] = classified["derivation"].astype("string")

    return classified


def _filter_as_of(
    statements: pd.DataFrame,
    as_of: str | pd.Timestamp | None,
) -> pd.DataFrame:
    if as_of is None:
        return statements

    if "filed_date" not in statements.columns:
        raise ValueError("statements must contain `filed_date` when as_of is used")

    cutoff = _normalize_timestamp(as_of)
    return statements.loc[
        statements["filed_date"].notna() & statements["filed_date"].le(cutoff)
    ].copy()


def _available_columns(frame: pd.DataFrame, candidates: tuple[str, ...]) -> list[str]:
    return [column_name for column_name in candidates if column_name in frame.columns]


def _deduplicate_filings(statements: pd.DataFrame) -> pd.DataFrame:
    valid = statements.dropna(
        subset=["ticker", "statement", "metric", "value", "period_end"]
    ).copy()

    if valid.empty:
        return valid

    identity_columns = _available_columns(valid, _IDENTITY_COLUMNS)
    sort_columns = list(identity_columns)

    for column_name in ("filed_date", "accession_number"):
        if column_name in valid.columns and column_name not in sort_columns:
            sort_columns.append(column_name)

    valid = valid.sort_values(sort_columns, na_position="first")
    return valid.drop_duplicates(subset=identity_columns, keep="last").reset_index(
        drop=True
    )


def _normalized_fiscal_period(row: pd.Series) -> str | None:
    fiscal_period = row.get("fiscal_period")
    normalized = str(fiscal_period).upper() if pd.notna(fiscal_period) else None
    frame = row.get("frame")

    if pd.notna(frame):
        frame_value = str(frame).upper()
        for quarter in ("Q1", "Q2", "Q3", "Q4"):
            if quarter in frame_value:
                return quarter

    return normalized


def _fiscal_year_series(frame: pd.DataFrame) -> pd.Series:
    period_end_year = frame["period_end"].dt.year.astype("Int64")

    if "fiscal_year" not in frame.columns:
        return period_end_year

    fiscal_year = pd.to_numeric(frame["fiscal_year"], errors="coerce").astype("Int64")
    return fiscal_year.fillna(period_end_year)


def _latest_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None

    sort_columns = ["period_end"]
    if "filed_date" in frame.columns:
        sort_columns.append("filed_date")

    return frame.sort_values(sort_columns, na_position="first").iloc[-1].copy()


def _source_for_period(
    frame: pd.DataFrame,
    period_type: str,
    fiscal_period: str | None = None,
) -> pd.Series | None:
    candidates = frame.loc[frame["period_type"].eq(period_type)].copy()

    if fiscal_period is not None:
        candidates = candidates.loc[
            candidates["_normalized_fiscal_period"].eq(fiscal_period)
        ]

    return _latest_row(candidates)


def _maximum_filed_date(rows: list[pd.Series]) -> pd.Timestamp | pd.NaT:
    filed_dates = [row.get("filed_date") for row in rows]
    valid_dates = [value for value in filed_dates if pd.notna(value)]
    return max(valid_dates) if valid_dates else pd.NaT


def _cumulative_from_direct(
    direct_rows: dict[str, pd.Series],
    through_quarter: int,
) -> dict[str, object] | None:
    quarter_names = [f"Q{quarter}" for quarter in range(1, through_quarter + 1)]

    if not all(quarter in direct_rows for quarter in quarter_names):
        return None

    rows = [direct_rows[quarter] for quarter in quarter_names]
    return {
        "value": float(sum(float(row["value"]) for row in rows)),
        "period_end": rows[-1]["period_end"],
        "filed_date": _maximum_filed_date(rows),
        "label": "+".join(quarter_names),
    }


def _cumulative_source(
    year_rows: pd.DataFrame,
    direct_rows: dict[str, pd.Series],
    fiscal_period: str,
) -> dict[str, object] | None:
    if fiscal_period == "Q1":
        direct = direct_rows.get("Q1")
        if direct is None:
            return None
        return {
            "value": float(direct["value"]),
            "period_end": direct["period_end"],
            "filed_date": direct.get("filed_date"),
            "label": "Q1",
        }

    ytd = _source_for_period(year_rows, "ytd", fiscal_period)
    if ytd is not None:
        return {
            "value": float(ytd["value"]),
            "period_end": ytd["period_end"],
            "filed_date": ytd.get("filed_date"),
            "label": f"YTD_{fiscal_period}",
        }

    return _cumulative_from_direct(direct_rows, int(fiscal_period[-1]))


def _derive_quarter(
    current: pd.Series,
    previous_cumulative: dict[str, object],
    fiscal_period: str,
) -> pd.Series:
    derived = current.copy()
    derived["value"] = float(current["value"]) - float(previous_cumulative["value"])
    derived["period_start"] = pd.Timestamp(
        previous_cumulative["period_end"]
    ) + pd.Timedelta(days=1)
    derived["fiscal_period"] = fiscal_period
    derived["period_type"] = "quarterly"
    derived["is_derived"] = True
    derived["derivation"] = (
        f"{current['period_type']}_{fiscal_period}-{previous_cumulative['label']}"
    )
    previous_filed_date = previous_cumulative.get("filed_date")
    filed_dates = [current.get("filed_date"), previous_filed_date]
    valid_filed_dates = [value for value in filed_dates if pd.notna(value)]
    derived["filed_date"] = max(valid_filed_dates) if valid_filed_dates else pd.NaT
    return derived


def _derive_discrete_quarters(statements: pd.DataFrame) -> pd.DataFrame:
    flows = statements.loc[
        statements["statement"].isin(_FLOW_STATEMENTS)
        & statements["period_type"].isin({"quarterly", "ytd", "annual"})
    ].copy()

    if flows.empty:
        return flows

    flows["_normalized_fiscal_period"] = flows.apply(
        _normalized_fiscal_period,
        axis=1,
    )
    flows["_derived_fiscal_year"] = _fiscal_year_series(flows)
    group_columns = _available_columns(flows, _METRIC_GROUP_COLUMNS)
    direct_output = flows.loc[flows["period_type"].eq("quarterly")].copy()
    direct_output["fiscal_period"] = direct_output["_normalized_fiscal_period"]
    derived_rows: list[pd.Series] = []

    for _, metric_rows in flows.groupby(group_columns, dropna=False, sort=False):
        for _, year_rows in metric_rows.groupby(
            "_derived_fiscal_year",
            dropna=False,
            sort=False,
        ):
            direct_rows: dict[str, pd.Series] = {}

            for quarter in ("Q1", "Q2", "Q3", "Q4"):
                direct = _source_for_period(year_rows, "quarterly", quarter)
                if direct is not None:
                    direct_rows[quarter] = direct

            for quarter, previous_quarter in (("Q2", "Q1"), ("Q3", "Q2")):
                if quarter in direct_rows:
                    continue

                current = _source_for_period(year_rows, "ytd", quarter)
                previous = _cumulative_source(
                    year_rows,
                    direct_rows,
                    previous_quarter,
                )

                if current is not None and previous is not None:
                    derived = _derive_quarter(current, previous, quarter)
                    derived_rows.append(derived)
                    direct_rows[quarter] = derived

            if "Q4" not in direct_rows:
                annual = _source_for_period(year_rows, "annual")
                previous = _cumulative_source(year_rows, direct_rows, "Q3")

                if annual is not None and previous is not None:
                    derived_rows.append(_derive_quarter(annual, previous, "Q4"))

    if derived_rows:
        derived_frame = pd.DataFrame(derived_rows)
        quarterly = pd.concat([direct_output, derived_frame], ignore_index=True)
    else:
        quarterly = direct_output

    helper_columns = ["_normalized_fiscal_period", "_derived_fiscal_year"]
    quarterly = quarterly.drop(columns=helper_columns, errors="ignore")
    return _deduplicate_filings(quarterly)


def _annual_instant_mask(statements: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=statements.index)

    if "fiscal_period" in statements.columns:
        mask |= statements["fiscal_period"].astype("string").str.upper().eq("FY")

    if "form" in statements.columns:
        mask |= statements["form"].astype("string").str.upper().isin(_ANNUAL_FORMS)

    if not mask.any():
        mask[:] = True

    return mask


def select_fundamental_periods(
    statements: StatementsInput,
    frequency: FiscalFrequency = "quarterly",
    as_of: str | pd.Timestamp | None = None,
    derive_quarters: bool = True,
) -> pd.DataFrame:
    """Select comparable fiscal observations without using future filings.

    Quarterly selection derives missing discrete Q2/Q3 values from YTD facts
    and Q4 from annual facts when the required cumulative observations exist.
    """
    if frequency not in {"all", "quarterly", "annual"}:
        raise ValueError("frequency must be `all`, `quarterly`, or `annual`")

    classified = classify_fundamental_periods(statements)
    available = _deduplicate_filings(_filter_as_of(classified, as_of=as_of))

    if frequency == "all" or available.empty:
        return available

    instant = available.loc[available["period_type"].eq("instant")].copy()

    if frequency == "quarterly":
        flows = (
            _derive_discrete_quarters(available)
            if derive_quarters
            else available.loc[available["period_type"].eq("quarterly")].copy()
        )
        selected = pd.concat([flows, instant], ignore_index=True, sort=False)
    else:
        flows = available.loc[available["period_type"].eq("annual")].copy()
        annual_instant = instant.loc[_annual_instant_mask(instant)].copy()
        selected = pd.concat([flows, annual_instant], ignore_index=True, sort=False)

    sort_columns = _available_columns(
        selected,
        ("ticker", "statement", "metric", "period_end", "filed_date"),
    )
    return selected.sort_values(sort_columns, na_position="first").reset_index(
        drop=True
    )


def _consecutive_quarter_window(window: pd.DataFrame) -> bool:
    period_ends = window["period_end"].sort_values()

    if period_ends.nunique() != 4:
        return False

    gaps = period_ends.diff().dropna().dt.days
    return bool(gaps.between(45, 125).all())


def _build_ttm_rows(quarterly: pd.DataFrame) -> pd.DataFrame:
    flows = quarterly.loc[
        quarterly["statement"].isin(_FLOW_STATEMENTS)
        & quarterly["period_type"].eq("quarterly")
    ].copy()

    if flows.empty:
        return flows

    group_columns = _available_columns(flows, _METRIC_GROUP_COLUMNS)
    ttm_rows: list[pd.Series] = []

    for _, metric_rows in flows.groupby(group_columns, dropna=False, sort=False):
        ordered = metric_rows.sort_values("period_end").reset_index(drop=True)

        for end_position in range(3, len(ordered)):
            window = ordered.iloc[end_position - 3 : end_position + 1]

            if not _consecutive_quarter_window(window):
                continue

            row = window.iloc[-1].copy()
            row["value"] = float(window["value"].sum())
            row["period_start"] = window.iloc[0].get("period_start")
            row["filed_date"] = _maximum_filed_date(
                [window.iloc[position] for position in range(len(window))]
            )
            row["fiscal_period"] = "TTM"
            row["period_type"] = "ttm"
            row["is_derived"] = True
            row["derivation"] = "sum_four_consecutive_quarters"

            for column_name in ("form", "frame", "accession_number"):
                if column_name in row.index:
                    row[column_name] = pd.NA

            ttm_rows.append(row)

    if not ttm_rows:
        return flows.iloc[0:0].copy()

    return _deduplicate_filings(pd.DataFrame(ttm_rows))


def _latest_balance_sheet(statements: pd.DataFrame) -> pd.DataFrame:
    balances = statements.loc[statements["period_type"].eq("instant")].copy()

    if balances.empty:
        return balances

    group_columns = _available_columns(
        balances,
        ("ticker", "metric", "unit", "provider", "taxonomy"),
    )
    sort_columns = group_columns + ["period_end"]
    if "filed_date" in balances.columns:
        sort_columns.append("filed_date")

    balances = balances.sort_values(sort_columns, na_position="first")
    return balances.groupby(group_columns, dropna=False, sort=False).tail(1)


def build_ttm_fundamentals(
    statements: StatementsInput,
    as_of: str | pd.Timestamp | None = None,
    include_latest_balance_sheet: bool = True,
) -> pd.DataFrame:
    """Build rolling TTM flow facts and optionally append latest balance facts."""
    all_periods = select_fundamental_periods(
        statements,
        frequency="all",
        as_of=as_of,
    )
    quarterly = select_fundamental_periods(
        all_periods,
        frequency="quarterly",
        derive_quarters=True,
    )
    ttm = _build_ttm_rows(quarterly)

    if not include_latest_balance_sheet:
        return ttm.reset_index(drop=True)

    latest_balance = _latest_balance_sheet(all_periods)
    combined = pd.concat([ttm, latest_balance], ignore_index=True, sort=False)
    sort_columns = _available_columns(
        combined,
        ("ticker", "statement", "metric", "period_end"),
    )

    if not sort_columns:
        return combined

    return combined.sort_values(sort_columns).reset_index(drop=True)


__all__ = [
    "FiscalFrequency",
    "StatementsInput",
    "build_ttm_fundamentals",
    "classify_fundamental_periods",
    "select_fundamental_periods",
]
