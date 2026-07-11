from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fintern.utils import detect_data_frequency, is_daily_or_finer

_TRADING_DAYS_PER_YEAR = 252.0
_TRADING_HOURS_PER_DAY = 6.5
_TRADING_MINUTES_PER_DAY = 390.0
_TRADING_SECONDS_PER_DAY = 23_400.0


@dataclass(frozen=True)
class Risk:
    """Represent risk metrics derived from price and OHLC market data."""

    ticker: str
    prices: pd.Series
    data: pd.DataFrame

    def _price_returns(self) -> pd.Series:
        if not isinstance(self.prices, pd.Series):
            raise TypeError("prices must be a pandas Series")

        prices = self.prices.copy()

        if prices.empty:
            raise ValueError("prices cannot be empty")

        if prices.isna().any():
            raise ValueError("prices cannot contain missing values")

        if (prices <= 0).any():
            raise ValueError("prices must be strictly positive")

        if isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.sort_index()

        returns = prices.astype(float).pct_change().dropna()

        if returns.empty:
            raise ValueError("At least two prices are required to compute returns")

        returns.name = self.ticker.strip().upper()
        return returns

    @staticmethod
    def _parse_frequency_string(freq: str) -> tuple[int, str]:
        for suffix in ("min", "s", "h", "d", "w", "m", "y"):
            if not freq.endswith(suffix):
                continue

            amount = freq.removesuffix(suffix)

            if amount.isdigit() and int(amount) > 0:
                return int(amount), suffix

            break

        raise ValueError(f"Unsupported frequency: {freq}")

    def _periods_per_year(self, periods_per_year: float | None = None) -> float:
        if periods_per_year is not None:
            if periods_per_year <= 0:
                raise ValueError("periods_per_year must be strictly positive")

            return float(periods_per_year)

        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise ValueError(
                "periods_per_year must be provided when prices do not use a "
                "DatetimeIndex"
            )

        frequency = detect_data_frequency(self.prices.sort_index())

        if frequency is None:
            raise ValueError(
                "Could not detect price frequency. Pass periods_per_year explicitly."
            )

        amount, unit = self._parse_frequency_string(frequency)

        if unit == "s":
            return _TRADING_DAYS_PER_YEAR * _TRADING_SECONDS_PER_DAY / amount

        if unit == "min":
            return _TRADING_DAYS_PER_YEAR * _TRADING_MINUTES_PER_DAY / amount

        if unit == "h":
            return _TRADING_DAYS_PER_YEAR * _TRADING_HOURS_PER_DAY / amount

        if unit == "d":
            return _TRADING_DAYS_PER_YEAR / amount

        if unit == "w":
            return 52.0 / amount

        if unit == "m":
            return 12.0 / amount

        return 1.0 / amount

    def volatility(self) -> float:
        """Calculate sample volatility from simple returns."""
        returns = self._price_returns()

        if len(returns) < 2:
            raise ValueError("At least two return observations are required")

        return float(returns.std())

    def annualized_volatility(self, periods_per_year: float | None = None) -> float:
        """Calculate annualized volatility from simple returns."""
        return float(
            self.volatility() * np.sqrt(self._periods_per_year(periods_per_year))
        )

    def rolling_volatility(self, window: int) -> pd.Series:
        """Calculate rolling sample volatility from simple returns."""
        if window <= 1:
            raise ValueError("window must be greater than 1")

        returns = self._price_returns()

        if len(returns) < window:
            raise ValueError("window must be smaller than or equal to observations")

        volatility = returns.rolling(window=window).std().dropna()
        volatility.index = pd.DatetimeIndex(volatility.index.to_numpy())
        volatility.index.name = None
        volatility.name = None

        return volatility

    def downside_deviation(self, target: float = 0.0) -> float:
        """Calculate downside deviation relative to a target return."""
        returns = self._price_returns()
        downside = np.minimum(returns - target, 0.0)

        return float(np.sqrt(np.mean(downside**2)))

    def historical_var(self, confidence_level: float = 0.95) -> float:
        """Calculate historical value at risk as a positive loss estimate."""
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be strictly between 0 and 1")

        returns = self._price_returns()
        quantile = returns.quantile(1 - confidence_level)

        return float(-quantile)

    def expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """Calculate historical expected shortfall as a positive loss estimate."""
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be strictly between 0 and 1")

        returns = self._price_returns()
        quantile = returns.quantile(1 - confidence_level)
        tail_losses = returns.loc[returns <= quantile]

        return float(-tail_losses.mean())

    def _ticker_high_low(self) -> pd.DataFrame:
        required_columns = {"date", "ticker", "high", "low"}
        missing_columns = sorted(required_columns - set(self.data.columns))

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"data must contain columns: {missing}")

        ticker_data = self.data.loc[
            self.data["ticker"].astype(str).str.upper() == self.ticker.strip().upper(),
            ["date", "high", "low"],
        ].copy()

        if ticker_data.empty:
            raise ValueError(f"No OHLC data found for ticker={self.ticker}")

        ticker_data["date"] = pd.to_datetime(ticker_data["date"])
        ticker_data = ticker_data.sort_values("date").set_index("date")

        ticker_data = ticker_data.astype(float)

        if (ticker_data["high"] <= 0).any() or (ticker_data["low"] <= 0).any():
            raise ValueError("high and low prices must be strictly positive")

        if (ticker_data["high"] < ticker_data["low"]).any():
            raise ValueError(
                "high prices must be greater than or equal to low prices"
            )

        return ticker_data
    

    def _ticker_ohlc(self) -> pd.DataFrame:
        required_columns = {"date", "ticker", "high", "low", "open", "close"}
        missing_columns = sorted(required_columns - set(self.data.columns))

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"data must contain columns: {missing}")

        ticker_data = self.data.loc[
            self.data["ticker"].astype(str).str.upper() == self.ticker.strip().upper(),
            ["date", "high", "low", "open", "close"],
        ].copy()

        if ticker_data.empty:
            raise ValueError(f"No OHLC data found for ticker={self.ticker}")

        ticker_data["date"] = pd.to_datetime(ticker_data["date"])
        ticker_data = ticker_data.sort_values("date").set_index("date")

        ticker_data = ticker_data.astype(float)

        if (
            (ticker_data["high"] <= 0).any()
            or (ticker_data["low"] <= 0).any()
            or (ticker_data["open"] <= 0).any()
            or (ticker_data["close"] <= 0).any()
        ):
            raise ValueError("prices must be strictly positive")

        if (ticker_data["high"] < ticker_data["low"]).any():
            raise ValueError(
                "high prices must be greater than or equal to low prices"
            )

        if (
            (ticker_data["open"] > ticker_data["high"]).any()
            or (ticker_data["open"] < ticker_data["low"]).any()
            or (ticker_data["close"] > ticker_data["high"]).any()
            or (ticker_data["close"] < ticker_data["low"]).any()
        ):
            raise ValueError(
                "open and close prices must lie between low and high"
            )

        return ticker_data

    def _daily_high_low(self) -> pd.DataFrame:
        ticker_data = self._ticker_high_low()

        detected_frequency = detect_data_frequency(ticker_data)

        if not is_daily_or_finer(detected_frequency):
            raise ValueError(
                "parkinson_volatility requires market data at daily frequency "
                "or finer"
            )

        daily_high_low = ticker_data.resample("B").agg({"high": "max", "low": "min"})
        daily_high_low = daily_high_low.dropna()

        if daily_high_low.empty:
            raise ValueError(f"No daily OHLC data found for ticker={self.ticker}")

        return daily_high_low

    def parkinson_volatility(self) -> float:
        """Calculate Parkinson volatility from daily high-low ranges."""
        daily_high_low = self._daily_high_low()
        log_ranges = np.log(daily_high_low["high"] / daily_high_low["low"])
        n = len(log_ranges)

        return float(np.sqrt(np.sum(log_ranges**2) / (4 * n * np.log(2))))

    def rolling_parkinson_volatility(self, window: int) -> pd.Series:
        """Calculate rolling Parkinson volatility on the native bar frequency."""
        if window <= 0:
            raise ValueError("window must be strictly positive")

        ticker_data = self._ticker_high_low()

        if len(ticker_data) < window:
            raise ValueError("window must be smaller than or equal to observations")

        log_ranges_squared = np.log(ticker_data["high"] / ticker_data["low"]) ** 2
        rolling_sum = log_ranges_squared.rolling(window=window).sum()
        volatility = np.sqrt(rolling_sum / (4 * window * np.log(2))).dropna()

        volatility.index = pd.DatetimeIndex(volatility.index.to_numpy())
        volatility.index.name = None

        return volatility
    

    def rolling_garmanKlass_volatility(self, window: int) -> pd.Series:
        """Calculate rolling Garman-Klass volatility on the native bar frequency."""
        if window <= 0:
            raise ValueError("window must be strictly positive")

        ticker_data = self._ticker_ohlc()

        if len(ticker_data) < window:
            raise ValueError("window must be smaller than or equal to observations")

        log_ranges_squared = np.log(ticker_data["high"] / ticker_data["low"]) ** 2
        log_open_close_squared = (
            np.log(ticker_data["close"] / ticker_data["open"]) ** 2
        )
        bar_variance = (
            0.5 * log_ranges_squared
            - (2 * np.log(2) - 1) * log_open_close_squared
        )

        rolling_variance = bar_variance.rolling(window=window).mean().clip(lower=0.0)
        volatility = np.sqrt(rolling_variance).dropna()

        volatility.index = pd.DatetimeIndex(volatility.index.to_numpy())
        volatility.index.name = None

        return volatility
