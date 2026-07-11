from pathlib import Path

import pandas as pd

from fintern.data import instruments as instruments_module


def test_resolve_instruments_saves_and_loads_flat_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    resolved_frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "ticker": ["AAPL", "MSFT"],
            "figi": ["BBG000B9XRY4", "BBG000BPH459"],
            "provider": ["openfigi", "openfigi"],
        }
    )

    class _DummyProvider:
        def resolve_instruments(self, symbols, exchange_code=None):
            del symbols, exchange_code
            return resolved_frame.copy()

    monkeypatch.setattr(
        instruments_module,
        "get_provider",
        lambda provider, capability: _DummyProvider(),
    )

    output_path = tmp_path / "instruments"
    resolved = instruments_module.resolve_instruments(
        symbols="AAPL MSFT",
        path=output_path,
        file_type="csv",
    )
    loaded = instruments_module.load_instruments(output_path)

    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        resolved.reset_index(drop=True),
    )
