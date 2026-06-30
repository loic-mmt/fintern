from dataclasses import dataclass

import pandas as pd
import numpy as np

@dataclass(frozen = True)
class Company:
    """Represent a company and its historical financial data."""

    ticker: str
    prices: pd.Series

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
        
        object.__setattr__(self, "ticker", normalized_ticker)
        object.__setattr__(self, "prices", self.prices.astype(float))

    def returns(self) -> pd.Series:
        """Calculate simple periodic returns."""
        return self.prices.pct_change().dropna()
    
    def total_return(self) -> float:
        """Calculate the total return over the complete period."""
        first_price = self.prices.iloc[0]
        last_price = self.prices.iloc[-1]

        return float(last_price / first_price - 1)
