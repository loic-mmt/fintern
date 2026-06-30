import pandas as pd
import pytest

from fintern.utils import (
    detect_data_frequency,
    is_daily_or_finer,
    resample_price_series,
)


def test_detect_data_frequency_from_hourly_series() -> None:
    index = pd.date_range("2025-01-01 09:00:00", periods=4, freq="h")
    series = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)

    detected = detect_data_frequency(series)

    assert detected == "1h"


def test_detect_data_frequency_from_business_day_data_frame() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-08",
                ]
            ),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    detected = detect_data_frequency(frame)

    assert detected == "1d"


def test_detect_data_frequency_from_monthly_datetime_index() -> None:
    index = pd.date_range("2025-01-31", periods=4, freq="ME")

    detected = detect_data_frequency(index)

    assert detected == "1m"


def test_detect_data_frequency_uses_date_column_name() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 09:00:00",
                    "2025-01-01 10:00:00",
                    "2025-01-01 11:00:00",
                ]
            ),
            "value": [1.0, 2.0, 3.0],
        }
    )

    detected = detect_data_frequency(frame, date_column="timestamp")

    assert detected == "1h"


def test_detect_data_frequency_rejects_missing_dates() -> None:
    frame = pd.DataFrame({"close": [100.0, 101.0]})

    with pytest.raises(
        ValueError,
        match="DatetimeIndex or contain requested date column",
    ):
        detect_data_frequency(frame)


def test_detect_data_frequency_rejects_single_timestamp() -> None:
    index = pd.DatetimeIndex([pd.Timestamp("2025-01-01")])

    with pytest.raises(ValueError, match="At least two timestamps"):
        detect_data_frequency(index)


def test_is_daily_or_finer_accepts_intraday_and_daily() -> None:
    assert is_daily_or_finer("30s")
    assert is_daily_or_finer("15min")
    assert is_daily_or_finer("3h")
    assert is_daily_or_finer("1d")


def test_is_daily_or_finer_rejects_coarser_than_daily() -> None:
    assert not is_daily_or_finer("2d")
    assert not is_daily_or_finer("1w")
    assert not is_daily_or_finer(None)


def test_resample_price_series_downsamples_business_days_to_weekly() -> None:
    index = pd.date_range("2025-01-06", periods=8, freq="B")
    series = pd.Series(range(100, 108), index=index, dtype=float)

    resampled = resample_price_series(series, "1w")

    expected = pd.Series(
        [104.0, 107.0],
        index=pd.to_datetime(["2025-01-10", "2025-01-17"]),
        dtype=float,
    )

    pd.testing.assert_series_equal(resampled, expected, check_freq=False)


def test_resample_price_series_downsamples_business_days_to_monthly() -> None:
    index = pd.to_datetime(["2025-01-30", "2025-01-31", "2025-02-03", "2025-02-27"])
    series = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)

    resampled = resample_price_series(series, "1m")

    expected = pd.Series(
        [101.0, 103.0],
        index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
    )

    pd.testing.assert_series_equal(resampled, expected, check_freq=False)


def test_resample_price_series_downsamples_hourly_to_daily() -> None:
    index = pd.to_datetime(
        [
            "2025-01-02 09:00:00",
            "2025-01-02 10:00:00",
            "2025-01-02 11:00:00",
            "2025-01-03 09:00:00",
            "2025-01-03 10:00:00",
            "2025-01-03 11:00:00",
        ]
    )
    series = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)

    resampled = resample_price_series(series, "1d")

    expected = pd.Series(
        [102.0, 105.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )

    pd.testing.assert_series_equal(resampled, expected, check_freq=False)


def test_resample_price_series_returns_copy_for_same_frequency() -> None:
    index = pd.date_range("2025-01-06", periods=4, freq="B")
    series = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)

    resampled = resample_price_series(series, "1d")

    pd.testing.assert_series_equal(resampled, series)
    assert resampled is not series


def test_resample_price_series_rejects_finer_frequency() -> None:
    index = pd.date_range("2025-01-06", periods=4, freq="B")
    series = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)

    with pytest.raises(ValueError, match="cannot be smaller"):
        resample_price_series(series, "1h")


def test_resample_price_series_rejects_non_datetime_index() -> None:
    series = pd.Series([100.0, 101.0, 102.0])

    with pytest.raises(ValueError, match="DatetimeIndex"):
        resample_price_series(series, "1d")
