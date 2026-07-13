import numpy as np
import pandas as pd
import pytest

from fintern.metrics.risk import Risk


def _make_risk(data: pd.DataFrame, ticker: str = "AAPL") -> Risk:
    prices = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    return Risk(ticker=ticker, prices=prices, data=data)


def _make_price_risk(
    prices: pd.Series,
    ticker: str = "AAPL",
) -> Risk:
    dates = (
        prices.index
        if isinstance(prices.index, pd.DatetimeIndex)
        else pd.date_range("2025-01-02", periods=len(prices), freq="B")
    )
    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": [ticker] * len(prices),
            "open": prices.to_numpy(),
            "high": prices.to_numpy(),
            "low": prices.to_numpy(),
            "close": prices.to_numpy(),
        }
    )
    return Risk(ticker=ticker, prices=prices, data=data)


def test_parkinson_volatility_from_daily_ohlc_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "high": [105.0, 108.0, 110.0],
            "low": [100.0, 102.0, 103.0],
        }
    )
    risk = _make_risk(data)

    log_ranges = np.log(data["high"] / data["low"])
    expected = float(np.sqrt(np.sum(log_ranges**2) / (4 * len(log_ranges) * np.log(2))))

    assert risk.parkinson_volatility() == pytest.approx(expected)


def test_parkinson_volatility_filters_to_requested_ticker() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "high": [105.0, 108.0, 110.0, 210.0, 220.0, 230.0],
            "low": [100.0, 102.0, 103.0, 200.0, 205.0, 210.0],
        }
    )
    risk = _make_risk(data, ticker="aapl")

    aapl = data.loc[data["ticker"] == "AAPL"]
    log_ranges = np.log(aapl["high"] / aapl["low"])
    expected = float(np.sqrt(np.sum(log_ranges**2) / (4 * len(log_ranges) * np.log(2))))

    assert risk.parkinson_volatility() == pytest.approx(expected)


def test_parkinson_volatility_aggregates_intraday_data_to_daily_ranges() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02 10:00:00",
                    "2025-01-02 15:00:00",
                    "2025-01-03 10:00:00",
                    "2025-01-03 15:00:00",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "high": [103.0, 105.0, 106.0, 108.0],
            "low": [100.0, 101.0, 102.0, 103.0],
        }
    )
    risk = _make_risk(data)

    daily_high = pd.Series([105.0, 108.0])
    daily_low = pd.Series([100.0, 102.0])
    log_ranges = np.log(daily_high / daily_low)
    expected = float(np.sqrt(np.sum(log_ranges**2) / (4 * len(log_ranges) * np.log(2))))

    assert risk.parkinson_volatility() == pytest.approx(expected)


def test_parkinson_volatility_rejects_weekly_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-17"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "high": [105.0, 108.0, 110.0],
            "low": [100.0, 102.0, 103.0],
        }
    )
    risk = _make_risk(data)

    with pytest.raises(ValueError, match="daily frequency or finer"):
        risk.parkinson_volatility()


def test_parkinson_volatility_rejects_non_positive_prices() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "high": [105.0, 108.0, 110.0],
            "low": [100.0, 0.0, 103.0],
        }
    )
    risk = _make_risk(data)

    with pytest.raises(ValueError, match="strictly positive"):
        risk.parkinson_volatility()


def test_parkinson_volatility_rejects_missing_columns() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "high": [105.0, 108.0, 110.0],
        }
    )
    risk = _make_risk(data)

    with pytest.raises(ValueError, match="data must contain columns: low"):
        risk.parkinson_volatility()


def test_rolling_parkinson_volatility_from_daily_ohlc_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "high": [105.0, 108.0, 110.0],
            "low": [100.0, 102.0, 103.0],
        }
    )
    risk = _make_risk(data)

    log_ranges_squared = np.log(data["high"] / data["low"]) ** 2
    expected = pd.Series(
        [
            np.sqrt(np.sum(log_ranges_squared.iloc[:2]) / (4 * 2 * np.log(2))),
            np.sqrt(np.sum(log_ranges_squared.iloc[1:3]) / (4 * 2 * np.log(2))),
        ],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(
        risk.rolling_parkinson_volatility(window=2),
        expected,
    )


def test_rolling_parkinson_volatility_from_intraday_ohlc_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02 10:00:00",
                    "2025-01-02 15:00:00",
                    "2025-01-03 10:00:00",
                    "2025-01-03 15:00:00",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "high": [103.0, 105.0, 106.0, 108.0],
            "low": [100.0, 101.0, 102.0, 103.0],
        }
    )
    risk = _make_risk(data)

    log_ranges_squared = np.log(data["high"] / data["low"]) ** 2
    expected = pd.Series(
        [
            np.sqrt(np.sum(log_ranges_squared.iloc[:2]) / (4 * 2 * np.log(2))),
            np.sqrt(np.sum(log_ranges_squared.iloc[1:3]) / (4 * 2 * np.log(2))),
            np.sqrt(np.sum(log_ranges_squared.iloc[2:4]) / (4 * 2 * np.log(2))),
        ],
        index=pd.to_datetime(
            [
                "2025-01-02 15:00:00",
                "2025-01-03 10:00:00",
                "2025-01-03 15:00:00",
            ]
        ),
    )

    pd.testing.assert_series_equal(
        risk.rolling_parkinson_volatility(window=2),
        expected,
    )


def test_rolling_parkinson_volatility_filters_to_requested_ticker() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "high": [105.0, 108.0, 110.0, 210.0, 220.0, 230.0],
            "low": [100.0, 102.0, 103.0, 200.0, 205.0, 210.0],
        }
    )
    risk = _make_risk(data, ticker="aapl")

    aapl = data.loc[data["ticker"] == "AAPL"]
    log_ranges_squared = np.log(aapl["high"] / aapl["low"]) ** 2
    expected = pd.Series(
        [
            np.sqrt(np.sum(log_ranges_squared.iloc[:2]) / (4 * 2 * np.log(2))),
            np.sqrt(np.sum(log_ranges_squared.iloc[1:3]) / (4 * 2 * np.log(2))),
        ],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(
        risk.rolling_parkinson_volatility(window=2),
        expected,
    )


def test_rolling_parkinson_volatility_rejects_non_positive_window() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "high": [105.0, 108.0, 110.0],
            "low": [100.0, 102.0, 103.0],
        }
    )
    risk = _make_risk(data)

    with pytest.raises(ValueError, match="window must be strictly positive"):
        risk.rolling_parkinson_volatility(window=0)


def test_rolling_parkinson_volatility_rejects_window_larger_than_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "ticker": ["AAPL", "AAPL"],
            "high": [105.0, 108.0],
            "low": [100.0, 102.0],
        }
    )
    risk = _make_risk(data)

    with pytest.raises(ValueError, match="smaller than or equal to observations"):
        risk.rolling_parkinson_volatility(window=3)


def test_rolling_garmanklass_volatility_from_daily_ohlc_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 104.0, 103.0],
            "high": [105.0, 108.0, 110.0],
            "low": [99.0, 102.0, 101.0],
            "close": [104.0, 103.0, 109.0],
        }
    )
    risk = _make_risk(data)

    bar_variance = (
        0.5 * np.log(data["high"] / data["low"]) ** 2
        - (2 * np.log(2) - 1) * np.log(data["close"] / data["open"]) ** 2
    )
    expected = pd.Series(
        [
            np.sqrt(bar_variance.iloc[:2].mean()),
            np.sqrt(bar_variance.iloc[1:3].mean()),
        ],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(
        risk.rolling_garmanKlass_volatility(window=2),
        expected,
    )
    pd.testing.assert_series_equal(
        risk.rolling_garman_klass_volatility(window=2),
        expected,
    )


def test_rolling_garmanklass_volatility_from_intraday_ohlc_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02 10:00:00",
                    "2025-01-02 15:00:00",
                    "2025-01-03 10:00:00",
                    "2025-01-03 15:00:00",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "open": [100.0, 102.0, 103.0, 104.0],
            "high": [103.0, 105.0, 106.0, 108.0],
            "low": [99.0, 101.0, 102.0, 103.0],
            "close": [102.0, 103.0, 104.0, 107.0],
        }
    )
    risk = _make_risk(data)

    bar_variance = (
        0.5 * np.log(data["high"] / data["low"]) ** 2
        - (2 * np.log(2) - 1) * np.log(data["close"] / data["open"]) ** 2
    )
    expected = pd.Series(
        [
            np.sqrt(bar_variance.iloc[:2].mean()),
            np.sqrt(bar_variance.iloc[1:3].mean()),
            np.sqrt(bar_variance.iloc[2:4].mean()),
        ],
        index=pd.to_datetime(
            [
                "2025-01-02 15:00:00",
                "2025-01-03 10:00:00",
                "2025-01-03 15:00:00",
            ]
        ),
    )

    pd.testing.assert_series_equal(
        risk.rolling_garmanKlass_volatility(window=2),
        expected,
    )


def test_rolling_garmanklass_volatility_filters_to_requested_ticker() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "open": [100.0, 104.0, 103.0, 200.0, 208.0, 210.0],
            "high": [105.0, 108.0, 110.0, 210.0, 220.0, 230.0],
            "low": [99.0, 102.0, 101.0, 198.0, 205.0, 209.0],
            "close": [104.0, 103.0, 109.0, 208.0, 210.0, 225.0],
        }
    )
    risk = _make_risk(data, ticker="aapl")

    aapl = data.loc[data["ticker"] == "AAPL"]
    bar_variance = (
        0.5 * np.log(aapl["high"] / aapl["low"]) ** 2
        - (2 * np.log(2) - 1) * np.log(aapl["close"] / aapl["open"]) ** 2
    )
    expected = pd.Series(
        [
            np.sqrt(bar_variance.iloc[:2].mean()),
            np.sqrt(bar_variance.iloc[1:3].mean()),
        ],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    pd.testing.assert_series_equal(
        risk.rolling_garmanKlass_volatility(window=2),
        expected,
    )


def test_rolling_garmanklass_volatility_rejects_non_positive_window() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 104.0, 103.0],
            "high": [105.0, 108.0, 110.0],
            "low": [99.0, 102.0, 101.0],
            "close": [104.0, 103.0, 109.0],
        }
    )
    risk = _make_risk(data)

    with pytest.raises(ValueError, match="window must be strictly positive"):
        risk.rolling_garmanKlass_volatility(window=0)


def test_rolling_garmanklass_volatility_rejects_window_larger_than_data() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "ticker": ["AAPL", "AAPL"],
            "open": [100.0, 104.0],
            "high": [105.0, 108.0],
            "low": [99.0, 102.0],
            "close": [104.0, 103.0],
        }
    )
    risk = _make_risk(data)

    with pytest.raises(ValueError, match="smaller than or equal to observations"):
        risk.rolling_garmanKlass_volatility(window=3)


def test_rolling_garmanklass_volatility_rejects_inconsistent_ohlc_bar() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "ticker": ["AAPL", "AAPL"],
            "open": [100.0, 109.0],
            "high": [105.0, 108.0],
            "low": [99.0, 102.0],
            "close": [104.0, 103.0],
        }
    )
    risk = _make_risk(data)

    with pytest.raises(ValueError, match="must lie between low and high"):
        risk.rolling_garmanKlass_volatility(window=2)


def test_volatility_from_simple_returns() -> None:
    prices = pd.Series(
        [100.0, 110.0, 99.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    risk = _make_price_risk(prices)
    expected = float(prices.pct_change().dropna().std())

    assert risk.volatility() == pytest.approx(expected)


def test_annualized_volatility_from_daily_prices() -> None:
    prices = pd.Series(
        [100.0, 110.0, 99.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    risk = _make_price_risk(prices)
    returns = prices.pct_change().dropna()
    expected = float(returns.std() * np.sqrt(252.0))

    assert risk.annualized_volatility() == pytest.approx(expected)


def test_rolling_volatility_from_simple_returns() -> None:
    prices = pd.Series(
        [100.0, 110.0, 99.0, 108.9],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]),
    )
    risk = _make_price_risk(prices)
    returns = prices.pct_change().dropna()
    expected = returns.rolling(window=2).std().dropna()
    expected.index = pd.DatetimeIndex(expected.index.to_numpy())
    expected.index.name = None

    pd.testing.assert_series_equal(risk.rolling_volatility(window=2), expected)


def test_downside_deviation_from_simple_returns() -> None:
    prices = pd.Series(
        [100.0, 110.0, 99.0, 103.95, 98.7525],
        index=pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        ),
    )
    risk = _make_price_risk(prices)
    returns = prices.pct_change().dropna()
    downside = np.minimum(returns, 0.0)
    expected = float(np.sqrt(np.mean(downside**2)))

    assert risk.downside_deviation() == pytest.approx(expected)


def test_historical_var_from_simple_returns() -> None:
    prices = pd.Series(
        [100.0, 90.0, 85.5, 87.21, 89.8263],
        index=pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        ),
    )
    risk = _make_price_risk(prices)
    returns = prices.pct_change().dropna()
    expected = float(-returns.quantile(0.25))

    assert risk.historical_var(confidence_level=0.75) == pytest.approx(expected)


def test_expected_shortfall_from_simple_returns() -> None:
    prices = pd.Series(
        [100.0, 90.0, 85.5, 87.21, 89.8263],
        index=pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        ),
    )
    risk = _make_price_risk(prices)
    returns = prices.pct_change().dropna()
    quantile = returns.quantile(0.25)
    expected = float(-returns.loc[returns <= quantile].mean())

    assert risk.expected_shortfall(confidence_level=0.75) == pytest.approx(expected)


def test_historical_var_rejects_invalid_confidence_level() -> None:
    prices = pd.Series(
        [100.0, 110.0, 99.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    risk = _make_price_risk(prices)

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        risk.historical_var(confidence_level=1.0)


def test_rolling_volatility_rejects_window_of_one() -> None:
    prices = pd.Series(
        [100.0, 110.0, 99.0],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    risk = _make_price_risk(prices)

    with pytest.raises(ValueError, match="greater than 1"):
        risk.rolling_volatility(window=1)
