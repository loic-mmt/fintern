import pandas as pd

import pytest

from fintern import Company

def test_ticker_is_normalized() -> None:
    prices = pd.Series([100.0, 110.0])

    company = Company(ticker = "aapl", prices = prices)

    assert company.ticker == "AAPL"


def test_total_return() -> None:
    prices = pd.Series([100.0, 110.0, 121.0])

    company = Company(ticker="AAPL", prices=prices)

    assert company.total_return() == pytest.approx(0.21)


def test_periodic_returns() -> None:

    prices = pd.Series([100.0, 110.0, 121.0])

    company = Company(ticker="AAPL", prices=prices)

    expected = pd.Series([0.10, 0.10], index=[1, 2])

    pd.testing.assert_series_equal(company.returns(), expected)


def test_empty_ticker_is_rejected() -> None:

    prices = pd.Series([100.0, 110.0])

    with pytest.raises(ValueError, match="ticker cannot be empty"):

        Company(ticker=" ", prices=prices)


def test_empty_prices_are_rejected() -> None:

    with pytest.raises(ValueError, match="prices cannot be empty"):

        Company(ticker="AAPL", prices=pd.Series(dtype=float))


def test_negative_prices_are_rejected() -> None:

    prices = pd.Series([100.0, -10.0])

    with pytest.raises(ValueError, match="strictly positive"):

        Company(ticker="AAPL", prices=prices)