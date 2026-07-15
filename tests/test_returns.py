import numpy as np
import pandas as pd
import pytest

from fintern.metrics.returns import Returns


def _make_returns() -> Returns:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "open": [101.0, 102.0, 104.0],
            "close": [100.0, 103.0, 105.0],
        }
    )
    prices = pd.Series(
        [100.0, 103.0, 105.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )

    return Returns(ticker="AAPL", prices=prices, data=data)


def test_overnight_returns_from_daily_ohlc_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "open": [101.0, 102.0, 104.0],
            "close": [100.0, 103.0, 105.0],
        }
    )
    prices = pd.Series(
        [100.0, 103.0, 105.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )

    returns = Returns(ticker="aapl", prices=prices, data=data)

    expected = pd.Series(
        [0.02, 104.0 / 103.0 - 1],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(returns.overnight_returns(), expected)


def test_overnight_returns_from_intraday_ohlc_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02 09:30:00",
                    "2025-01-02 16:00:00",
                    "2025-01-03 09:30:00",
                    "2025-01-03 16:00:00",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "open": [101.0, 102.0, 103.0, 104.0],
            "close": [101.5, 100.0, 103.5, 102.0],
        }
    )
    prices = pd.Series(
        [100.0, 102.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )

    returns = Returns(ticker="AAPL", prices=prices, data=data)

    expected = pd.Series([0.03], index=pd.to_datetime(["2025-01-03"]))

    pd.testing.assert_series_equal(returns.overnight_returns(), expected)


def test_overnight_returns_rejects_weekly_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-17"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 110.0, 120.0],
            "close": [105.0, 115.0, 125.0],
        }
    )
    prices = pd.Series(
        [105.0, 115.0, 125.0],
        index=pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-17"]),
    )

    returns = Returns(ticker="AAPL", prices=prices, data=data)

    with pytest.raises(ValueError, match="daily frequency or finer"):
        returns.overnight_returns()


def test_intraday_returns_from_daily_ohlc_data() -> None:
    returns = _make_returns()

    expected = pd.Series(
        [100.0 / 101.0 - 1, 103.0 / 102.0 - 1, 105.0 / 104.0 - 1],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(returns.intraday_returns(), expected)


def test_lagged_returns_uses_requested_lag_in_column_name_and_values() -> None:
    returns = _make_returns()

    result = returns.lagged_returns(lag=1)

    expected = pd.DataFrame(
        {
            "return": [0.03, 105.0 / 103.0 - 1],
            "return_lag_1": [float("nan"), 0.03],
        },
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_lagged_returns_rejects_non_positive_lag() -> None:
    returns = _make_returns()

    with pytest.raises(ValueError, match="lag must be strictly positive"):
        returns.lagged_returns(lag=0)


def test_log_returns() -> None:
    returns = _make_returns()

    expected = pd.Series(
        [np.log(103.0 / 100.0), np.log(105.0 / 103.0)],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(returns.log_returns(), expected)


def test_cummulative_returns() -> None:
    returns = _make_returns()

    expected = pd.Series(
        [0.03, 0.05],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(returns.cummulative_returns(), expected)


def test_cumulative_returns_uses_canonical_name() -> None:
    returns = _make_returns()

    pd.testing.assert_series_equal(
        returns.cumulative_returns(),
        returns.cummulative_returns(),
    )


def test_holding_period_return() -> None:
    returns = _make_returns()

    result = returns.holding_period_return("2025-01-02", "2025-01-06")

    assert result == pytest.approx(0.05)


def test_holding_period_return_requires_at_least_two_prices() -> None:
    returns = _make_returns()

    with pytest.raises(ValueError, match="at least two prices"):
        returns.holding_period_return("2025-01-02", "2025-01-02")


def test_rolling_returns() -> None:
    returns = _make_returns()

    result = returns.rolling_returns(window=2)
    expected = pd.Series(
        [0.05],
        index=pd.to_datetime(["2025-01-06"]),
    )

    pd.testing.assert_series_equal(result, expected)


def test_wealth_index() -> None:
    returns = _make_returns()

    expected = pd.Series(
        [103.0, 105.0],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(returns.wealth_index(), expected)


def test_excess_returns_with_float_benchmark() -> None:
    returns = _make_returns()

    expected = pd.Series(
        [0.01, 105.0 / 103.0 - 1 - 0.02],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(returns.excess_returns(0.02), expected)


def test_excess_returns_uses_canonical_name() -> None:
    returns = _make_returns()

    pd.testing.assert_series_equal(
        returns.excess_returns(0.02),
        returns.exces_returns(0.02),
    )


def test_simple_to_log_returns() -> None:
    returns = _make_returns()

    expected = pd.Series(
        [np.log1p(0.03), np.log1p(105.0 / 103.0 - 1)],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(returns.simple_to_log_returns(), expected)


def test_log_to_simple_returns() -> None:
    returns = _make_returns()

    expected = returns.returns()

    pd.testing.assert_series_equal(returns.log_to_simple_returns(), expected)


def test_cagr() -> None:
    returns = _make_returns()

    expected = float((105.0 / 100.0) ** (1 / (3 / 252)) - 1)

    assert returns.CAGR() == pytest.approx(expected)
    assert returns.cagr() == pytest.approx(expected)


def test_forward_returns() -> None:
    returns = _make_returns()

    expected = pd.Series(
        [0.03, 105.0 / 103.0 - 1],
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )

    pd.testing.assert_series_equal(returns.forward_returns(), expected)


def test_forward_returns_rejects_non_positive_periods() -> None:
    returns = _make_returns()

    with pytest.raises(ValueError, match="periods must be strictly positive"):
        returns.forward_returns(periods=0)
