from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir


def save_json(data: dict[str, Any] | list[Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    try:
        df.to_parquet(p, index=False)
        return p
    except ImportError:
        csv_path = p.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return _read_existing_table(p)
    csv_fallback = p.with_suffix(".csv")
    if csv_fallback.exists():
        return _read_existing_table(csv_fallback)
    raise FileNotFoundError(f"Table artifact not found: {p}")


def _read_existing_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=["date"], infer_datetime_format=True)
    raise ValueError(f"Unsupported table artifact format: {path.suffix}")


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)
