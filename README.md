# Fintern

Fintern is an open-source Python library for explainable financial analysis.

It aims to transform market and fundamental data into:

- financial metrics;
- historical comparisons;
- peer comparisons;
- red-flag detection;
- transparent company scores;
- analyst-style reports.

## Installation

```bash
pip install fintern
```

Fintern is currently under active development and is not yet published on PyPI.

## Development installation

```bash
git clone https://github.com/loic-mmt/fintern.git
cd fintern

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

## Example

```python
import pandas as pd

from fintern import Company

prices = pd.Series([100.0, 105.0, 110.0])

company = Company("AAPL", prices)

print(company.total_return())
```

## Run tests

```bash
pytest
```

## Disclaimer

Fintern is intended for research and educational purposes. It does not provide investment advice.