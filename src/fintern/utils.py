from __future__ import annotations

from typing import Literal

import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


def select_price_series(
    data: pd.DataFrame,
    ticker: str,
    column: str = "adj_close",
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    normalized_ticker = ticker.strip().upper()

    if "date" not in data.columns:
        raise ValueError("data must contain a `date` column")

    if "ticker" not in data.columns:
        raise ValueError("data must contain a `ticker` column")

    if column not in data.columns:
        raise ValueError(f"data must contain `{column}` column")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    series = (
        frame.loc[frame["ticker"].str.upper() == normalized_ticker, ["date", column]]
        .dropna()
        .sort_values("date")
        .set_index("date")[column]
    )

    if start is not None:
        series = series.loc[pd.Timestamp(start) :]

    if end is not None:
        series = series.loc[: pd.Timestamp(end)]

    if series.empty:
        raise ValueError(
            f"No data found for ticker={normalized_ticker} and column={column}"
        )

    series.name = normalized_ticker
    return series.astype(float)


def _extract_datetime_index(
    data: pd.Series | pd.DataFrame | pd.DatetimeIndex,
    date_column: str,
) -> pd.DatetimeIndex:
    if isinstance(data, pd.DatetimeIndex):
        datetime_index = data
    elif isinstance(data, pd.Series):
        if isinstance(data.index, pd.DatetimeIndex):
            datetime_index = data.index
        elif pd.api.types.is_datetime64_any_dtype(data):
            datetime_index = pd.DatetimeIndex(data.dropna())
        else:
            raise ValueError("Series must have DatetimeIndex or datetime-like values.")
    elif isinstance(data, pd.DataFrame):
        if date_column in data.columns:
            datetime_index = pd.DatetimeIndex(pd.to_datetime(data[date_column]))
        elif isinstance(data.index, pd.DatetimeIndex):
            datetime_index = data.index
        else:
            raise ValueError(
                "DataFrame must have DatetimeIndex or contain requested date column."
            )
    else:
        raise TypeError("data must be a pandas Series, DataFrame, or DatetimeIndex")

    normalized_index = pd.DatetimeIndex(datetime_index).dropna().sort_values().unique()

    if len(normalized_index) < 2:
        raise ValueError("At least two timestamps are required to detect frequency.")

    return normalized_index


def _normalize_offset_frequency(offset) -> str | None:
    if isinstance(offset, (offsets.BusinessHour, offsets.Hour)):
        return f"{offset.n}h"

    if isinstance(offset, (offsets.BDay, offsets.Day)):
        return f"{offset.n}d"

    if isinstance(offset, offsets.Week):
        return f"{offset.n}w"

    if isinstance(
        offset,
        (
            offsets.BMonthBegin,
            offsets.BMonthEnd,
            offsets.MonthBegin,
            offsets.MonthEnd,
        ),
    ):
        return f"{offset.n}m"

    if isinstance(
        offset,
        (
            offsets.BQuarterBegin,
            offsets.BQuarterEnd,
            offsets.QuarterBegin,
            offsets.QuarterEnd,
        ),
    ):
        return f"{offset.n * 3}m"

    if isinstance(
        offset,
        (
            offsets.BYearBegin,
            offsets.BYearEnd,
            offsets.YearBegin,
            offsets.YearEnd,
        ),
    ):
        return f"{offset.n}y"

    return None


def _normalize_timedelta_frequency(delta: pd.Timedelta) -> str | None:
    total_seconds = int(delta.total_seconds())

    if total_seconds <= 0:
        return None

    minute_seconds = 60
    hour_seconds = 60 * minute_seconds
    day_seconds = 24 * hour_seconds
    week_seconds = 7 * day_seconds

    if total_seconds % week_seconds == 0:
        return f"{total_seconds // week_seconds}w"

    if total_seconds % day_seconds == 0:
        return f"{total_seconds // day_seconds}d"

    if total_seconds % hour_seconds == 0:
        return f"{total_seconds // hour_seconds}h"

    if total_seconds % minute_seconds == 0:
        return f"{total_seconds // minute_seconds}min"

    return f"{total_seconds}s"


def detect_data_frequency(
    data: pd.Series | pd.DataFrame | pd.DatetimeIndex,
    date_column: str = "date",
) -> str | None:
    datetime_index = _extract_datetime_index(data, date_column=date_column)

    if len(datetime_index) >= 3:
        inferred_frequency = pd.infer_freq(datetime_index)

        if inferred_frequency is not None:
            normalized_frequency = _normalize_offset_frequency(
                to_offset(inferred_frequency)
            )

            if normalized_frequency is not None:
                return normalized_frequency

    deltas = datetime_index.to_series().diff().dropna()

    if deltas.empty:
        return None

    mode_deltas = deltas.mode()

    if not mode_deltas.empty:
        mode_frequency = _normalize_timedelta_frequency(mode_deltas.iloc[0])

        if mode_frequency is not None:
            return mode_frequency

    median_delta = deltas.median()
    return _normalize_timedelta_frequency(median_delta)


_RESAMPLE_RULES = {
    "1h": "h",
    "1d": "B",
    "1w": "W-FRI",
    "1m": "ME",
    "3m": "QE",
    "1y": "YE",
}

_FREQUENCY_RANKS = {
    "s": 0,
    "min": 1,
    "h": 2,
    "d": 3,
    "w": 4,
    "m": 5,
    "y": 6,
}


def _parse_frequency_string(freq: str) -> tuple[int, int, str]:
    for suffix in ("min", "s", "h", "d", "w", "m", "y"):
        if not freq.endswith(suffix):
            continue

        amount = freq.removesuffix(suffix)

        if amount.isdigit() and int(amount) > 0:
            return _FREQUENCY_RANKS[suffix], int(amount), suffix

        break

    raise ValueError(f"Unsupported frequency: {freq}")


def is_daily_or_finer(freq: str | None) -> bool:
    if freq is None:
        return False

    _, amount, unit = _parse_frequency_string(freq)

    if unit in {"s", "min", "h"}:
        return True

    return unit == "d" and amount == 1


def resample_price_series(
    series: pd.Series,
    freq: Literal["1h", "1d", "1w", "1m", "3m", "1y"],
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")

    sorted_series = series.sort_index()
    current_frequency = detect_data_frequency(sorted_series)

    if current_frequency is None:
        raise ValueError("Could not detect current frequency.")

    current_rank, current_amount, current_unit = _parse_frequency_string(
        current_frequency
    )
    requested_rank, requested_amount, requested_unit = _parse_frequency_string(freq)

    if requested_rank < current_rank:
        raise ValueError(
            "The requested frequency cannot be smaller than the current frequency."
        )

    if requested_rank == current_rank and requested_amount < current_amount:
        raise ValueError(
            "The requested frequency cannot be smaller than the current frequency."
        )

    if (
        requested_rank == current_rank
        and requested_amount == current_amount
        and requested_unit == current_unit
    ):
        return sorted_series.copy()

    return sorted_series.resample(_RESAMPLE_RULES[freq]).last().dropna()


