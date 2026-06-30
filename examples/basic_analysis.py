import pandas as pd

from fintern import Company


prices = pd.Series(
    [100.0, 102.0, 99.0, 105.0, 110.0],
    name="price",
)

company = Company(
    ticker="AAPL",
    prices=prices,
)

print(f"Ticker: {company.ticker}")
print(f"Total return: {company.total_return():.2%}")
print(company.returns())