from fintern.analysis import (
    HistoricalAnalysis,
    RedFlag,
    RedFlagAnalysis,
    RedFlagThresholds,
)
from fintern.company import Company
from fintern.data import (
    FundamentalsData,
    MarketData,
    build_company_dataset,
    build_ttm_fundamentals,
    classify_fundamental_periods,
    download_fundamentals,
    download_market_data,
    load_fundamentals,
    load_instruments,
    load_market_data,
    resolve_instruments,
    select_fundamental_periods,
)
from fintern.metrics import Growth, Profitability, Returns, Risk, Valuation

__all__ = [
    "Company",
    "FundamentalsData",
    "Growth",
    "HistoricalAnalysis",
    "MarketData",
    "Profitability",
    "RedFlag",
    "RedFlagAnalysis",
    "RedFlagThresholds",
    "Returns",
    "Risk",
    "Valuation",
    "build_company_dataset",
    "build_ttm_fundamentals",
    "classify_fundamental_periods",
    "download_fundamentals",
    "download_market_data",
    "load_fundamentals",
    "load_instruments",
    "load_market_data",
    "resolve_instruments",
    "select_fundamental_periods",
]
__version__ = "0.1.0"
