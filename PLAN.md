# Data Layer Rebuild Plan

## Summary
- Split user-facing data API into `market` and `fundamentals`. Best for simplicity, less ambiguity, easier docs.
- Do **not** offer naive concatenated `DataFrame` as default. Market data = time series by trading date. Fundamentals = statement periods by filing/report date. Different grain. Default return separate tables.
- Add explicit join helper for advanced use. It should perform keyed/as-of merge, not raw concat.
- Implement v1 provider core around:
  - `Yahoo` for market prices
  - `SEC` for filings/fundamentals
  - `OpenFIGI` for instrument mapping
- Keep `FMP`, `EODHD`, `Alpha Vantage` as optional adapters behind same interface, scaffolded now, not feature-complete in v1.

## Architecture Changes
- Refactor [market.py](/home/teneviaone/Documents/fintern/src/fintern/data/market.py) into orchestration layer only.
- Keep file/folder load/save logic in `market.py`, but move provider download logic out of it.
- Add provider abstraction in `providers/base.py`:
  - market capability method
  - fundamentals capability method
  - instrument lookup capability method
  - provider availability check: dependency present, API key present, feature supported
- Add provider registry file. Best new file:
  - `data/providers/registry.py`
  - responsibility: discover provider classes, choose first available provider, validate requested provider, expose friendly errors
- Use `exceptions.py` for custom errors:
  - `ProviderNotAvailableError`
  - `MissingDependencyError`
  - `MissingAPIKeyError`
  - `NoProviderConfiguredError`
  - `UnsupportedCapabilityError`
  - `InstrumentResolutionError`
- Use `instruments.py` for normalized identifier layer:
  - ticker
  - exchange
  - currency
  - FIGI
  - CIK
  - ISIN if available
  - provider source metadata
- Use `fundamentals.py` for normalized statement outputs:
  - income statement
  - balance sheet
  - cash flow
  - company/profile metadata
  - standardized columns + normalized dates/periods
- Add one more file for shared schemas/models. Best new file:
  - `data/models.py`
  - dataclasses or typed aliases for provider result metadata and normalized outputs
- Keep provider modules thin:
  - `yahoo.py`: market price fetch only in v1
  - `sec.py`: CIK/company facts/submissions fundamentals in v1
  - `openfigi.py`: symbol-to-identifier resolution in v1
  - `fmp.py`, `eodhd.py`, `alpha_vantage.py`: scaffold class, availability checks, normalized interface, unimplemented capabilities allowed where needed

## Public API
- User-facing market API:
  - `download_market_data(...)`
  - `load_market_data(...)`
- User-facing fundamentals API:
  - `download_fundamentals(...)`
  - `load_fundamentals(...)`
- User-facing advanced join helper:
  - `build_company_dataset(market_data, fundamentals, join="asof" | "period_end")`
- Default provider behavior:
  - if user passes no provider, registry selects first available provider supporting requested capability
  - if requested provider missing dependency or API key, raise clear custom error with exact install/env guidance
  - if no provider available at all, raise one combined error listing:
    - missing dependencies
    - missing API keys
    - unsupported capability per provider
- Do not force all dependencies on user.
- Add optional extras in `pyproject.toml`:
  - `yahoo`
  - `sec`
  - `openfigi`
  - `fmp`
  - `eodhd`
  - `alpha-vantage`
  - `data` aggregate extra
- Environment variable convention:
  - one env var per provider API key
  - SEC/OpenFIGI user-agent/header config documented if required
- Save/load behavior:
  - market datasets: keep current partitioned parquet/CSV support
  - fundamentals datasets: save partitioned by `ticker`, `statement`, `fiscal_year` or `period`
  - instrument mappings: save as flat table
- Join recommendation:
  - market and fundamentals separate by default
  - combined dataset only through helper
  - helper should use filing/report period logic, not raw row concat

## Testing
- Core tests use committed fixtures, never live APIs by default.
- Add `tests/fixtures/data/providers/...` with normalized snapshots:
  - market fixture from Yahoo-style normalized output
  - SEC company facts / filings fixture
  - OpenFIGI mapping fixture
- Unit tests should mock provider clients/parsers, not hit network.
- Integration tests should read fixture payloads and verify normalization + orchestration.
- Live tests should exist but be opt-in only:
  - pytest marker like `live_api`
  - skipped unless env vars + explicit flag present
- No CI dependence on rate-limited providers.
- For one-time download workflow:
  - create small script or test utility to refresh fixtures manually
  - fixture refresh never runs during normal test suite

## Important Implementation Decisions
- `market.py` should no longer import provider SDKs directly.
- Provider modules own optional imports. Missing import handled inside provider availability check.
- Normalization must happen provider-side or in provider-specific parser, before data reaches public API.
- `OpenFIGI` should not be treated as price/fundamentals provider. It is identifier-resolution provider.
- `SEC` should not be treated as market-price provider. Fundamentals/provider roles stay explicit.
- `Yahoo` should not be fundamentals source in v1, even if some metadata exists. Keep scope clean.
- `fundamentals` output should use long, normalized, analysis-friendly tables rather than provider-native JSON trees.

## Test Cases
- no provider requested + one available provider -> succeeds
- requested provider missing dependency -> clear custom error
- requested provider missing API key -> clear custom error
- no providers available -> combined explanation error
- market download via registry -> normalized OHLCV table
- fundamentals download via registry -> normalized statement tables
- OpenFIGI symbol lookup -> normalized instrument table
- save/load round-trip for market datasets
- save/load round-trip for fundamentals datasets
- advanced dataset builder performs documented join behavior
- fixture-based tests run with zero optional provider packages installed
- live tests skip cleanly when env vars/deps missing

## Assumptions
- v1 supports scalable architecture first, not full implementation of every provider capability.
- Separate market/fundamentals API is best default for user simplicity.
- Combined market+fundamentals output should be explicit helper only.
- Optional dependency model is mandatory.
- Provider files may stay present but partially implemented if registry reports capability truthfully.
