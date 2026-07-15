from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fintern.utils import detect_data_frequency, is_daily_or_finer


@dataclass(frozen=True)
class Returns:
    """Represent return metrics derived from price and OHLC market data."""

    ticker: str
    prices: pd.Series
    data: pd.DataFrame

    def __post_init__(self) -> None:
        normalized_ticker = self.ticker.strip().upper()

        if not normalized_ticker:
            raise ValueError("ticker cannot be empty")

        if self.prices.empty:
            raise ValueError("prices cannot be empty")

        if self.prices.isna().any():
            raise ValueError("prices cannot contain missing values")

        if (self.prices <= 0).any():
            raise ValueError("prices must be strictly positive")

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if self.data.empty:
            raise ValueError("data cannot be empty")

        object.__setattr__(self, "ticker", normalized_ticker)
        object.__setattr__(self, "prices", self.prices.astype(float))

    def returns(self) -> pd.Series:
        """Calculate simple periodic returns."""
        return self.prices.pct_change().dropna()

    def total_return(self) -> float:
        """Calculate total return over complete period."""
        first_price = self.prices.iloc[0]
        last_price = self.prices.iloc[-1]

        return float(last_price / first_price - 1)

    def _daily_open_close(self) -> pd.DataFrame:
        required_columns = {"date", "ticker", "open", "close"}
        missing_columns = sorted(required_columns - set(self.data.columns))

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"data must contain columns: {missing}")

        ticker_data = self.data.loc[
            self.data["ticker"].astype(str).str.upper() == self.ticker,
            ["date", "open", "close"],
        ].copy()

        if ticker_data.empty:
            raise ValueError(f"No OHLC data found for ticker={self.ticker}")

        ticker_data["date"] = pd.to_datetime(ticker_data["date"])
        ticker_data = ticker_data.sort_values("date").set_index("date")

        detected_frequency = detect_data_frequency(ticker_data)

        if not is_daily_or_finer(detected_frequency):
            raise ValueError(
                "overnight_returns requires market data at daily frequency or finer"
            )

        daily_ohlc = ticker_data.resample("B").agg({"open": "first", "close": "last"})
        daily_ohlc = daily_ohlc.dropna()

        if daily_ohlc.empty:
            raise ValueError(f"No daily OHLC data found for ticker={self.ticker}")

        return daily_ohlc.astype(float)

    def overnight_returns(self) -> pd.Series:
        """Calculate overnight returns from previous close to next open."""
        daily_ohlc = self._daily_open_close()
        overnight = daily_ohlc["open"] / daily_ohlc["close"].shift(1) - 1

        overnight = overnight.dropna()
        overnight.index = pd.DatetimeIndex(overnight.index.to_numpy())
        overnight.index.name = None

        return overnight

    def intraday_returns(self) -> pd.Series:
        """Calculate same-day returns from open to close."""
        daily_ohlc = self._daily_open_close()
        intraday = daily_ohlc["close"] / daily_ohlc["open"] - 1

        intraday.index = pd.DatetimeIndex(intraday.index.to_numpy())
        intraday.index.name = None

        return intraday

    def lagged_returns(self, lag: int = 1) -> pd.DataFrame:
        if lag <= 0:
            raise ValueError("lag must be strictly positive")

        returns = self.returns()
        df = pd.DataFrame({"return": returns})
        df[f"return_lag_{lag}"] = df["return"].shift(lag)

        return df

    def log_returns(self) -> pd.Series:
        """Calculate logarithmic returns"""
        return np.log(self.prices / self.prices.shift(1)).dropna()

    def cumulative_returns(self) -> pd.Series:
        """Calculate cumulative compounded returns."""
        return (1 + self.returns()).cumprod() - 1

    def cummulative_returns(self) -> pd.Series:
        """Compatibility alias for :meth:`cumulative_returns`."""
        return self.cumulative_returns()

    def holding_period_return(self, start: str, end: str) -> float:
        """Calculate total return over a specified holding period."""
        prices = self.prices.copy()
        prices.index = pd.to_datetime(prices.index)
        period_prices = prices.loc[start:end]

        if len(period_prices) < 2:
            raise ValueError("holding period must contain at least two prices")

        return float(period_prices.iloc[-1] / period_prices.iloc[0] - 1)

    def rolling_returns(self, window: int) -> pd.Series:
        """Calculate rolling returns over a fixed window."""
        if window <= 0:
            raise ValueError("window must be strictly positive")

        if len(self.prices) <= window:
            raise ValueError("window must be smaller than the number of prices")

        return self.prices.div(self.prices.shift(window)).sub(1).dropna()

    def wealth_index(self, initial_value: float = 100) -> pd.Series:
        """Create a wealth index from periodic returns."""
        if initial_value <= 0:
            raise ValueError("initial_value must be strictly positive")

        return initial_value * (1 + self.returns()).cumprod()

    def excess_returns(self, benchmark_returns: pd.Series | float) -> pd.Series:
        """Calculate returns in excess of a benchmark or fixed rate."""
        return self.returns().subtract(benchmark_returns).dropna()

    def exces_returns(self, benchmark_returns: pd.Series | float) -> pd.Series:
        """Compatibility alias for :meth:`excess_returns`."""
        return self.excess_returns(benchmark_returns)

    def simple_to_log_returns(self) -> pd.Series:
        """Convert simple returns into logarithmic returns."""
        simple_returns = self.returns()

        if (simple_returns <= -1).any():
            raise ValueError("simple returns must be greater than -1")

        return np.log1p(simple_returns)

    def log_to_simple_returns(self) -> pd.Series:
        """Convert logarithmic returns into simple returns."""
        return np.expm1(self.log_returns())

    def cagr(self) -> float:
        """Calculate the compound annual growth rate."""
        total_days = len(self.prices)
        number_of_years = total_days / 252
        first_price = self.prices.iloc[0]
        last_price = self.prices.iloc[-1]
        return float((last_price / first_price) ** (1 / number_of_years) - 1)

    def CAGR(self) -> float:
        """Compatibility alias for :meth:`cagr`."""
        return self.cagr()

    def forward_returns(self, periods: int = 1) -> pd.Series:
        """Calculate future returns over a specified horizon."""
        if periods <= 0:
            raise ValueError("periods must be strictly positive")

        return self.prices.shift(-periods).div(self.prices).sub(1).dropna()
