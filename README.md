<p align="center">
  <img src="./Bandeau%20Fintern.png" alt="Fintern banner" />
</p>

# Fintern

Fintern is an open-source Python toolkit for explainable financial analysis.
It brings market data, fundamentals, instrument identifiers, and transparent
analytics together in one consistent workflow.

Fintern is designed for analysts, researchers, and developers who want financial
calculations they can inspect, understand, and extend. Data provenance, joins,
and analytical logic remain explicit at every step.

## Capabilities

- normalized market data downloads, storage, and retrieval
- normalized financial statements and company profiles
- instrument resolution across financial identifiers
- explicit market and fundamentals alignment
- fiscal-period normalization and trailing-twelve-month fundamentals
- company, return, risk, growth, profitability, and valuation analytics
- historical analysis and explainable red-flag detection
- extensible providers and metrics architecture

The roadmap expands these foundations into peer comparisons, transparent
scoring, valuation workflows, and analyst-style reporting.

## Installation

Install the latest development version directly from source.

### Core install

```bash
git clone https://github.com/loic-mmt/fintern.git
cd fintern

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -e .
```

### With development tools

```bash
python -m pip install -e ".[dev]"
```

### With data provider extras

```bash
python -m pip install -e ".[data]"
```

Available extras:

| Extra | Purpose |
| --- | --- |
| `yahoo` | Yahoo Finance market data via `yfinance` |
| `sec` | SEC fundamentals via `requests` |
| `openfigi` | OpenFIGI instrument mapping via `requests` |
| `fmp` | Scaffolded adapter |
| `eodhd` | Scaffolded adapter |
| `alpha-vantage` | Scaffolded adapter |
| `data` | Convenience extra for current v1 data stack |
| `dev` | Testing, linting, and packaging tools |

## Quickstart

### Company analysis

```python
import pandas as pd

from fintern import Company

prices = pd.Series([100.0, 105.0, 110.0])
company = Company("AAPL", prices)

print(company.returns())
print(company.total_return())
```

### Market data

```python
from fintern import download_market_data

market = download_market_data(
    tickers=["AAPL", "MSFT"],
    start="2025-01-01",
    end="2025-03-31",
    provider="yahoo",
)

print(market.head())
```

### Fundamentals data

```python
from fintern import download_fundamentals

fundamentals = download_fundamentals(
    tickers="AAPL",
    provider="sec",
)

print(fundamentals["statements"].head())
print(fundamentals["company_profile"].head())
```

### Build a combined dataset explicitly

```python
from fintern import (
    build_company_dataset,
    download_fundamentals,
    download_market_data,
)

market = download_market_data(
    "AAPL",
    start="2025-01-01",
    end="2025-06-30",
    provider="yahoo",
)
fundamentals = download_fundamentals("AAPL", provider="sec")

dataset = build_company_dataset(
    market_data=market,
    fundamentals=fundamentals,
    join="asof",
)

print(dataset.head())
```

### Normalize fiscal periods and build TTM fundamentals

```python
from fintern import build_ttm_fundamentals, select_fundamental_periods

quarterly = select_fundamental_periods(
    fundamentals,
    frequency="quarterly",
    as_of="2025-06-30",
)
ttm = build_ttm_fundamentals(
    fundamentals,
    as_of="2025-06-30",
)
```

Quarterly selection separates discrete quarters from YTD and annual facts,
keeps the latest filing available at `as_of`, and derives missing discrete
quarters when sufficient cumulative facts exist.

## Data Providers

Provider roles are intentionally explicit:

- `Yahoo`
  Market prices only.
- `SEC`
  Fundamentals only.
- `OpenFIGI`
  Instrument resolution only.

This separation keeps data provenance clear and lets each source serve the
capability it handles best.

## Provider Configuration

Depending on the provider, you may need optional dependencies and environment
variables.

Common variables:

- `FINTERN_OPENFIGI_API_KEY`
- `FINTERN_OPENFIGI_USER_AGENT`
- `FINTERN_SEC_USER_AGENT`
- `FINTERN_FMP_API_KEY`
- `FINTERN_EODHD_API_KEY`
- `FINTERN_ALPHA_VANTAGE_API_KEY`

If a requested provider cannot be used, Fintern raises a clear error explaining
whether the issue is:

- missing dependency
- missing API key / environment variable
- unsupported capability
- no compatible provider available

## Storage Model

Fintern supports both in-memory use and save/load workflows.

- market datasets can be saved as single files or partitioned datasets
- fundamentals are stored as a multi-table dataset
- instrument mappings are stored as a flat table

By default, the library keeps:

- market data separate from fundamentals
- fundamentals separate from instrument mapping

If you want a combined analysis table, use `build_company_dataset(...)`
explicitly rather than a naive row-wise concat.

## Documentation

- [Data Layer Architecture](./docs/data-layer.md)

## Development

Run the test suite:

```bash
pytest -p no:cacheprovider
```

Run linting:

```bash
ruff check src tests
```

Some tests are intentionally optional:

- live API tests are opt-in only
- parquet tests are skipped when `pyarrow` is not installed

## Status

Fintern is actively developed around a modular data and analytics architecture.
Its public API may evolve as new providers, metrics, and analysis workflows are
added.

## Disclaimer

Fintern is intended for research and educational purposes. It does not provide
investment advice.
