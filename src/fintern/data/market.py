from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

_SUPPORTED_FILE_TYPES = {
    ".csv": "csv",
    ".csv.gz": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".txt": "text",
    ".tsv": "text",
    ".nc": "netcdf",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".feather": "feather",
}
_IGNORED_DATASET_FILES = {"_SUCCESS", "_metadata", "_common_metadata"}
_YFINANCE_PRICE_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


def _normalize_path(path: str | Path) -> Path:
    normalized = Path(path).expanduser()

    if not normalized.exists():
        raise FileNotFoundError(f"File/Folder not found: {normalized}")

    return normalized


def _detect_path_format(path: str | Path) -> str:
    path = _normalize_path(path)

    if path.is_file():
        return "file"

    if path.is_dir():
        return "folder"

    raise FileNotFoundError(f"File/Folder not found: {path}")


def _detect_file_type(path_type, path: str | Path) -> str:
    path = _normalize_path(path)

    if path_type != "file":
        raise ValueError("Directory given instead of file path.")

    suffixes = [suffix.lower() for suffix in path.suffixes]

    for start_index in range(len(suffixes)):
        candidate = "".join(suffixes[start_index:])
        if candidate in _SUPPORTED_FILE_TYPES:
            return _SUPPORTED_FILE_TYPES[candidate]

    return "unknown"


def is_year_folder(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.isdigit()
        and len(path.name) == 4
        and 1900 <= int(path.name) <= 2100
    )


def _classify_folder(path: Path) -> str:
    if is_year_folder(path):
        return "year"

    if "=" in path.name:
        key, _, value = path.name.partition("=")

        if key == "year" and value.isdigit() and len(value) == 4:
            return "year"

        return key or "partition"

    return "ticker"


def _is_hidden_relative_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _iter_supported_files(path: Path) -> list[Path]:
    files: list[Path] = []

    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue

        relative_path = item.relative_to(path)

        if _is_hidden_relative_path(relative_path):
            continue

        if item.name in _IGNORED_DATASET_FILES:
            continue

        if _detect_file_type("file", item) != "unknown":
            files.append(item)

    return files


def _detect_folder_arrangement(path_type, file_type: str, path: str | Path):
    del file_type

    path = _normalize_path(path)

    if path_type != "folder":
        raise ValueError("File given instead of directory path.")

    files: list[str] = []
    folders: list[str] = []

    for item in sorted(path.rglob("*")):
        relative_path = item.relative_to(path)

        if _is_hidden_relative_path(relative_path):
            continue

        if item.is_dir():
            folders.append(_classify_folder(item))
            continue

        if item.name in _IGNORED_DATASET_FILES:
            continue

        detected_type = _detect_file_type("file", item)

        if detected_type != "unknown":
            files.append(detected_type)

    return sorted(set(files)), sorted(set(folders))


@dataclass(frozen=True)
class _MarketPathDescription:
    path: Path
    path_type: str
    file_type: str | None
    leaf_files: tuple[Path, ...]
    file_types: tuple[str, ...]


def _describe_market_path(path: str | Path) -> _MarketPathDescription:
    normalized_path = _normalize_path(path)
    path_type = _detect_path_format(normalized_path)

    if path_type == "file":
        file_type = _detect_file_type(path_type, normalized_path)

        if file_type == "unknown":
            suffix_or_name = normalized_path.suffix or normalized_path.name
            raise ValueError(f"Unsupported file type: {suffix_or_name}")

        return _MarketPathDescription(
            path=normalized_path,
            path_type=path_type,
            file_type=file_type,
            leaf_files=(normalized_path,),
            file_types=(file_type,),
        )

    leaf_files = tuple(_iter_supported_files(normalized_path))

    if not leaf_files:
        raise ValueError(f"No supported files found under folder: {normalized_path}")

    file_types = tuple(
        sorted({_detect_file_type("file", file_path) for file_path in leaf_files})
    )
    single_type = file_types[0] if len(file_types) == 1 else None

    return _MarketPathDescription(
        path=normalized_path,
        path_type=path_type,
        file_type=single_type,
        leaf_files=leaf_files,
        file_types=file_types,
    )


def _coerce_to_dataframe(data: Any, source: Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name or "value")

    if isinstance(data, dict):
        values = list(data.values())
        is_scalar_dict = all(
            not isinstance(value, (dict, list, tuple, pd.Series, pd.DataFrame))
            for value in values
        )

        if is_scalar_dict:
            return pd.DataFrame([data])

        return pd.json_normalize(data, sep=".")

    if isinstance(data, (list, tuple)):
        if not data:
            return pd.DataFrame()

        if all(not isinstance(item, (dict, list, tuple)) for item in data):
            return pd.DataFrame({"value": list(data)})

        return pd.DataFrame(data)

    if data is None:
        return pd.DataFrame()

    raise TypeError(
        f"Unsupported payload type {type(data).__name__} loaded from {source}"
    )


def _read_json_file(path: Path) -> pd.DataFrame:
    for kwargs in ({"lines": True}, {}):
        try:
            return _coerce_to_dataframe(pd.read_json(path, **kwargs), path)
        except ValueError:
            continue

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    return _coerce_to_dataframe(payload, path)


def _read_text_file(path: Path) -> pd.DataFrame:
    csv_kwargs = {"engine": "python"}

    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")

    try:
        return pd.read_csv(path, sep=None, **csv_kwargs)
    except (pd.errors.ParserError, UnicodeDecodeError):
        contents = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return pd.DataFrame({"value": contents})


def _read_hdf5_file(path: Path) -> pd.DataFrame:
    try:
        return _coerce_to_dataframe(pd.read_hdf(path), path)
    except (KeyError, ValueError) as exc:
        with pd.HDFStore(path) as store:
            keys = store.keys()

        if not keys:
            raise ValueError(f"No datasets found in HDF5 file: {path}") from exc

        frames = []

        for key in keys:
            frame = _coerce_to_dataframe(pd.read_hdf(path, key=key), path)
            frame.insert(0, "hdf_key", key.lstrip("/"))
            frames.append(frame)

        return pd.concat(frames, ignore_index=True, sort=False)


def _read_netcdf_file(path: Path) -> pd.DataFrame:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Reading .nc files requires optional dependency `xarray`."
        ) from exc

    dataset = xr.open_dataset(path)

    try:
        return dataset.to_dataframe().reset_index()
    finally:
        dataset.close()


def _read_single_market_file(path: Path) -> pd.DataFrame:
    file_type = _detect_file_type("file", path)

    if file_type == "csv":
        return pd.read_csv(path)

    if file_type == "excel":
        return pd.read_excel(path)

    if file_type == "parquet":
        return pd.read_parquet(path)

    if file_type == "json":
        return _read_json_file(path)

    if file_type == "text":
        return _read_text_file(path)

    if file_type == "netcdf":
        return _read_netcdf_file(path)

    if file_type == "hdf5":
        return _read_hdf5_file(path)

    if file_type == "pickle":
        return _coerce_to_dataframe(pd.read_pickle(path), path)

    if file_type == "feather":
        return pd.read_feather(path)

    raise ValueError(f"Unsupported file type: {path}")


def _add_source_path(
    frame: pd.DataFrame,
    source_path: Path,
    root_path: Path,
) -> pd.DataFrame:
    sourced_frame = frame.copy()
    column_name = "source_path"

    while column_name in sourced_frame.columns:
        column_name = f"_{column_name}"

    sourced_frame.insert(0, column_name, source_path.relative_to(root_path).as_posix())

    return sourced_frame


def _coerce_partition_value(value: str) -> int | str:
    if value.isdigit():
        return int(value)

    return value


def _extract_partition_pairs(
    source_path: Path,
    root_path: Path,
) -> list[tuple[str, int | str]]:
    partitions: list[tuple[str, int | str]] = []
    relative_path = source_path.relative_to(root_path)

    for folder_name in relative_path.parts[:-1]:
        if "=" not in folder_name:
            continue

        column_name, _, raw_value = folder_name.partition("=")

        if not column_name:
            continue

        partitions.append((column_name, _coerce_partition_value(raw_value)))

    return partitions


def _add_partition_columns(
    frame: pd.DataFrame,
    source_path: Path,
    root_path: Path,
) -> pd.DataFrame:
    partitioned_frame = _add_source_path(frame, source_path, root_path)
    insert_at = 1

    for column_name, value in _extract_partition_pairs(source_path, root_path):
        if column_name in partitioned_frame.columns:
            column = partitioned_frame[column_name]

            if column.eq(value).all():
                continue

            column_name = f"partition_{column_name}"

        while column_name in partitioned_frame.columns:
            column_name = f"_{column_name}"

        partitioned_frame.insert(insert_at, column_name, value)
        insert_at += 1

    return partitioned_frame


def _normalize_tickers(tickers: str | Sequence[str]) -> list[str]:
    if isinstance(tickers, str):
        raw_tickers = tickers.replace(",", " ").split()
    else:
        raw_tickers = [str(ticker) for ticker in tickers]

    normalized_tickers = [
        ticker.strip().upper() for ticker in raw_tickers if ticker.strip()
    ]

    if not normalized_tickers:
        raise ValueError("tickers cannot be empty")

    return list(dict.fromkeys(normalized_tickers))


def _download_raw_market_data(
    tickers: Sequence[str],
    start: str | None,
    end: str | None,
    interval: str,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Downloading market data requires optional dependency `yfinance`."
        ) from exc

    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        progress=False,
    )


def _normalize_column_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _normalize_downloaded_market_data(
    raw_data: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    if raw_data.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )

    normalized = raw_data.copy()
    normalized.index = pd.to_datetime(normalized.index)
    index_name = normalized.index.name or "date"

    if isinstance(normalized.columns, pd.MultiIndex):
        first_level = {str(value) for value in normalized.columns.get_level_values(0)}
        second_level = {str(value) for value in normalized.columns.get_level_values(1)}

        if _YFINANCE_PRICE_FIELDS & first_level:
            normalized = normalized.stack(level=1, future_stack=True)
        elif _YFINANCE_PRICE_FIELDS & second_level:
            normalized = normalized.stack(level=0, future_stack=True)
        else:
            raise ValueError("Unexpected yfinance download format.")

        normalized = normalized.rename_axis(index=[index_name, "ticker"]).reset_index()
    else:
        normalized = normalized.reset_index()
        normalized["ticker"] = tickers[0]

    normalized = normalized.rename(columns={index_name: "date"})
    normalized = normalized.rename(columns=_normalize_column_name)
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()

    ordered_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    present_ordered_columns = [
        column_name
        for column_name in ordered_columns
        if column_name in normalized.columns
    ]
    remaining_columns = [
        column_name
        for column_name in normalized.columns
        if column_name not in present_ordered_columns
    ]

    return normalized[present_ordered_columns + remaining_columns].sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def _prepare_partitioned_market_data(
    data: pd.DataFrame,
    partition_cols: Sequence[str],
) -> pd.DataFrame:
    partitioned_data = data.copy()
    missing_partition_cols = [
        column_name
        for column_name in partition_cols
        if column_name not in partitioned_data.columns
    ]

    if "year" in missing_partition_cols:
        if "date" not in partitioned_data.columns:
            raise ValueError("`year` partition requires `date` column.")

        partitioned_data["year"] = (
            pd.to_datetime(partitioned_data["date"]).dt.year.astype("int64")
        )
        missing_partition_cols.remove("year")

    if missing_partition_cols:
        missing = ", ".join(missing_partition_cols)
        raise ValueError(f"Missing partition columns: {missing}")

    return partitioned_data


def _is_csv_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith(".csv") or lower_name.endswith(".csv.gz")


def _is_parquet_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith(".parquet") or lower_name.endswith(".pq")


def _normalize_output_path(
    path: str | Path,
    file_type: Literal["csv", "parquet"],
) -> Path:
    normalized_path = Path(path).expanduser()

    if (
        file_type == "csv"
        and normalized_path.suffixes
        and not _is_csv_path(normalized_path)
    ):
        raise ValueError("CSV output path must end with `.csv` or `.csv.gz`.")

    if file_type == "parquet" and normalized_path.suffixes and not _is_parquet_path(
        normalized_path
    ):
        raise ValueError("Parquet output path must end with `.parquet` or `.pq`.")

    return normalized_path


def _save_to_csv(
    data: pd.DataFrame,
    path: str | Path,
    partition_cols: Sequence[str] = ("ticker", "year"),
) -> Path:
    output_path = _normalize_output_path(path, "csv")

    if _is_csv_path(output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        return output_path

    partitioned_data = _prepare_partitioned_market_data(data, partition_cols)
    output_path.mkdir(parents=True, exist_ok=True)

    grouped_data = partitioned_data.groupby(list(partition_cols), sort=True)

    for partition_key, group in grouped_data:
        partition_values = (
            partition_key if isinstance(partition_key, tuple) else (partition_key,)
        )
        partition_path = output_path

        for column_name, value in zip(partition_cols, partition_values, strict=True):
            partition_path = partition_path / f"{column_name}={value}"

        partition_path.mkdir(parents=True, exist_ok=True)
        group.drop(columns=list(partition_cols)).to_csv(
            partition_path / "data.csv",
            index=False,
        )

    return output_path


def _save_to_parquet(
    data: pd.DataFrame,
    path: str | Path,
    partition_cols: Sequence[str] = ("ticker", "year"),
) -> Path:
    output_path = _normalize_output_path(path, "parquet")

    if _is_parquet_path(output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(output_path, index=False)
        return output_path

    partitioned_data = _prepare_partitioned_market_data(data, partition_cols)
    output_path.mkdir(parents=True, exist_ok=True)
    partitioned_data.to_parquet(
        output_path,
        index=False,
        partition_cols=list(partition_cols),
    )

    return output_path


def _load_folder_market_data(path: Path) -> pd.DataFrame | dict[str, pd.DataFrame]:
    description = _describe_market_path(path)

    if set(description.file_types) == {"parquet"}:
        try:
            return pd.read_parquet(description.path)
        except (ImportError, OSError, ValueError):
            pass

    loaded_files: dict[str, pd.DataFrame] = {}

    for file_path in description.leaf_files:
        relative_path = file_path.relative_to(description.path).as_posix()
        loaded_files[relative_path] = _read_single_market_file(file_path)

    if len(description.file_types) == 1:
        frames = [
            _add_partition_columns(
                frame,
                description.path / relative_path,
                description.path,
            )
            for relative_path, frame in loaded_files.items()
        ]
        return pd.concat(frames, ignore_index=True, sort=False)

    return loaded_files


class MarketData:
    """
    Load or download market data.
    """

    @staticmethod
    def load_market_data(path: str | Path) -> pd.DataFrame | dict[str, pd.DataFrame]:
        description = _describe_market_path(path)

        if description.path_type == "file":
            return _read_single_market_file(description.path)

        return _load_folder_market_data(description.path)


    @staticmethod
    def download_market_data(
        tickers: str | Sequence[str],
        start: str | None = None,
        end: str | None = None,
        path: str | Path | None = None,
        file_type: Literal["csv", "parquet"] = "parquet",
        interval: str = "1d",
    ) -> pd.DataFrame:
        normalized_tickers = _normalize_tickers(tickers)
        raw_data = _download_raw_market_data(
            tickers=normalized_tickers,
            start=start,
            end=end,
            interval=interval,
        )
        data = _normalize_downloaded_market_data(raw_data, normalized_tickers)

        if path is None:
            return data

        if file_type == "csv":
            _save_to_csv(data=data, path=path)
            return data

        _save_to_parquet(data=data, path=path)
        return data
