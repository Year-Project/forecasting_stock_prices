from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import load_json, load_table, save_json
from .mlflow_model import MODEL_BUNDLE_ARTIFACT, StockReturnPyFuncRouter, safe_filename
from .strict_protocol import run_strict_per_ticker_protocol


HORIZON_SPECS = {
    "week": {"horizon": 5, "registered_model_name": "stock_return_forecaster_week"},
    "month": {"horizon": 21, "registered_model_name": "stock_return_forecaster_month"},
}
DEFAULT_EXPERIMENT_NAME = "stock_return_forecasting"
DEFAULT_ALIAS = "prd"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact_root = Path(args.artifact_root).resolve()

    if args.tracking_uri:
        _set_mlflow_tracking_uri(args.tracking_uri)

    for horizon_name in args.horizons:
        if horizon_name not in HORIZON_SPECS:
            raise ValueError(f"Unsupported horizon: {horizon_name}")
        if not args.skip_training:
            train_horizon(
                artifact_root=artifact_root,
                horizon_name=horizon_name,
                force_retrain=args.force_retrain,
                n_trials=args.n_trials,
                include_lstm=args.include_lstm,
            )
        write_error_robustness_report(artifact_root, horizon_name)
        write_reproducibility_manifest(artifact_root, horizon_name)
        if not args.skip_register:
            log_and_register_horizon(
                artifact_root=artifact_root,
                horizon_name=horizon_name,
                experiment_name=args.experiment_name,
                alias=args.alias,
            )
    return 0


def train_horizon(
    artifact_root: Path,
    horizon_name: str,
    force_retrain: bool = False,
    n_trials: int = 12,
    include_lstm: bool = False,
) -> dict[str, Any]:
    spec = HORIZON_SPECS[horizon_name]
    horizon = int(spec["horizon"])
    data_dir = artifact_root / "data" / "horizons" / horizon_name
    feature_payload = load_json(data_dir / "feature_columns.json")
    model_df = load_table(data_dir / "model_dataset.parquet")
    model_df["date"] = pd.to_datetime(model_df["date"])
    feature_cols = list(feature_payload["feature_columns"])
    target_col = str(feature_payload.get("target_column", f"target_return_{horizon}_next_open"))

    return run_strict_per_ticker_protocol(
        model_df=model_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_configs=make_default_model_configs(feature_cols, n_trials=n_trials, include_lstm=include_lstm),
        artifact_dir=artifact_root / "horizons" / horizon_name,
        force_retrain=force_retrain,
        primary_metric="directional_accuracy",
        random_state=42,
        run_metadata={"horizon_name": horizon_name, "horizon": horizon},
    )


def make_default_model_configs(
    feature_cols: list[str],
    n_trials: int = 12,
    include_lstm: bool = False,
) -> list[dict[str, Any]]:
    momentum_columns = [
        col
        for col in [
            "rolling_ret_mean_5",
            "rolling_ret_mean_10",
            "rolling_ret_mean_20",
            "residual_momentum_5",
            "residual_momentum_20",
        ]
        if col in feature_cols
    ]
    configs: list[dict[str, Any]] = [
        {
            "name": "naive_persistence",
            "model_type": "naive_persistence",
            "static_params": {},
            "search_space": {},
        },
        {
            "name": "momentum",
            "model_type": "momentum",
            "static_params": {"column": momentum_columns[0] if momentum_columns else "ret_1"},
            "search_space": {"column": {"type": "categorical", "choices": momentum_columns or ["ret_1"]}},
            "n_trials": min(n_trials, max(1, len(momentum_columns) or 1)),
        },
        {
            "name": "ridge",
            "model_type": "ridge",
            "static_params": {},
            "search_space": {"alpha": {"type": "categorical", "choices": [0.1, 1.0, 10.0, 100.0]}},
            "n_trials": min(n_trials, 4),
        },
        {
            "name": "hist_gradient_boosting",
            "model_type": "hist_gradient_boosting",
            "static_params": {},
            "search_space": {
                "learning_rate": {"type": "categorical", "choices": [0.03, 0.06, 0.1]},
                "max_leaf_nodes": {"type": "categorical", "choices": [15, 31]},
            },
            "n_trials": min(n_trials, 6),
        },
        {
            "name": "lightgbm",
            "model_type": "lightgbm",
            "static_params": {},
            "search_space": {
                "learning_rate": {"type": "categorical", "choices": [0.03, 0.06, 0.1]},
                "num_leaves": {"type": "categorical", "choices": [15, 31, 63]},
            },
            "n_trials": min(n_trials, 6),
        },
        {
            "name": "xgboost",
            "model_type": "xgboost",
            "static_params": {},
            "search_space": {
                "learning_rate": {"type": "categorical", "choices": [0.03, 0.06, 0.1]},
                "max_depth": {"type": "categorical", "choices": [2, 3, 4]},
            },
            "n_trials": min(n_trials, 6),
        },
        {
            "name": "catboost",
            "model_type": "catboost",
            "static_params": {},
            "search_space": {
                "learning_rate": {"type": "categorical", "choices": [0.03, 0.06, 0.1]},
                "depth": {"type": "categorical", "choices": [3, 4, 5]},
            },
            "n_trials": min(n_trials, 6),
        },
    ]
    if include_lstm and importlib.util.find_spec("torch") is not None:
        configs.append(
            {
                "name": "lstm",
                "model_type": "lstm",
                "static_params": {
                    "lookback": 60,
                    "hidden_size": 64,
                    "num_layers": 1,
                    "head_dropout": 0.1,
                    "max_epochs": 80,
                    "patience": 10,
                    "device": "auto",
                },
                "search_space": {},
                "input_mode": "full_frame",
                "feature_cols": feature_cols,
            }
        )
    return configs


def log_and_register_horizon(
    artifact_root: Path,
    horizon_name: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    alias: str = DEFAULT_ALIAS,
) -> str:
    import mlflow
    from mlflow.tracking import MlflowClient

    spec = HORIZON_SPECS[horizon_name]
    registered_model_name = str(spec["registered_model_name"])
    strict_root = artifact_root / "horizons" / horizon_name / "strict_protocol"
    reports_dir = strict_root / "reports"
    bundle_dir = build_model_bundle(artifact_root, horizon_name)
    manifest_path = write_reproducibility_manifest(artifact_root, horizon_name)
    robustness_paths = write_error_robustness_report(artifact_root, horizon_name)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{horizon_name}-strict-prd") as run:
        mlflow.set_tags(
            {
                "horizon_name": horizon_name,
                "horizon": str(spec["horizon"]),
                "registry_alias": alias,
                "training_protocol": "strict_per_ticker",
            }
        )
        mlflow.log_params({"horizon_name": horizon_name, "horizon": int(spec["horizon"])})
        _log_report_metrics(reports_dir)
        mlflow.log_artifact(str(manifest_path), artifact_path="reproducibility")
        for path in robustness_paths:
            mlflow.log_artifact(str(path), artifact_path="reports")
        if reports_dir.exists():
            mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
        plots_dir = strict_root / "plots"
        if plots_dir.exists():
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=StockReturnPyFuncRouter(),
            artifacts={MODEL_BUNDLE_ARTIFACT: str(bundle_dir)},
            code_paths=[str(Path(__file__).resolve().parents[1])],
            pip_requirements=_model_pip_requirements(),
        )
        registered = mlflow.register_model(model_info.model_uri, registered_model_name)
        client = MlflowClient()
        client.set_registered_model_alias(registered_model_name, alias, registered.version)
        client.set_model_version_tag(registered_model_name, registered.version, "run_id", run.info.run_id)
        client.set_model_version_tag(registered_model_name, registered.version, "horizon_name", horizon_name)
        client.set_model_version_tag(registered_model_name, registered.version, "horizon", str(spec["horizon"]))
        return f"models:/{registered_model_name}@{alias}"


def build_model_bundle(artifact_root: Path, horizon_name: str) -> Path:
    spec = HORIZON_SPECS[horizon_name]
    horizon_root = artifact_root / "horizons" / horizon_name
    strict_root = horizon_root / "strict_protocol"
    reports_dir = strict_root / "reports"
    bundle_dir = horizon_root / "mlflow_model_bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "models").mkdir(parents=True, exist_ok=True)

    selected = load_table(reports_dir / "selected_models_by_ticker.parquet")
    if selected.empty:
        raise ValueError(f"No selected models found for horizon {horizon_name}")
    feature_payload = load_json(artifact_root / "data" / "horizons" / horizon_name / "feature_columns.json")

    selected_records = []
    for row in selected.to_dict(orient="records"):
        ticker = str(row["ticker"])
        model_name = str(row["model_name"])
        source = strict_root / "models" / safe_filename(model_name) / safe_filename(ticker) / "final.pkl"
        if not source.exists():
            raise FileNotFoundError(f"Selected model payload not found: {source}")
        target = bundle_dir / "models" / safe_filename(model_name) / safe_filename(ticker) / "final.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        selected_records.append(_json_safe_record(row))

    save_json(selected_records, bundle_dir / "selected_models.json")
    save_json(feature_payload, bundle_dir / "feature_columns.json")
    save_json(
        {
            "horizon_name": horizon_name,
            "horizon": int(spec["horizon"]),
            "registered_model_name": spec["registered_model_name"],
            "model_count": len(selected_records),
        },
        bundle_dir / "metadata.json",
    )
    return bundle_dir


def write_reproducibility_manifest(artifact_root: Path, horizon_name: str) -> Path:
    spec = HORIZON_SPECS[horizon_name]
    reports_dir = artifact_root / "horizons" / horizon_name / "strict_protocol" / "reports"
    manifest = {
        "horizon_name": horizon_name,
        "horizon": int(spec["horizon"]),
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "data_artifacts": _artifact_hashes(artifact_root / "data" / "horizons" / horizon_name),
        "report_artifacts": _artifact_hashes(reports_dir),
        "dependencies": _dependency_snapshot(),
    }
    path = reports_dir / "reproducibility_manifest.json"
    save_json(manifest, path)
    return path


def write_error_robustness_report(artifact_root: Path, horizon_name: str) -> list[Path]:
    reports_dir = artifact_root / "horizons" / horizon_name / "strict_protocol" / "reports"
    selected = _load_optional_table(reports_dir / "selected_models_by_ticker.parquet")
    validation = _load_optional_table(reports_dir / "validation_prediction_metrics.parquet")
    test = _load_optional_table(reports_dir / "test_prediction_metrics.parquet")
    signal = _load_optional_table(reports_dir / "test_signal_metrics.parquet")
    leakage = _load_optional_table(reports_dir / "leakage_audit.parquet")

    payload = {
        "horizon_name": horizon_name,
        "selected_models": selected.to_dict(orient="records") if not selected.empty else [],
        "validation_summary": _numeric_summary(validation),
        "test_summary": _numeric_summary(test),
        "signal_summary": _numeric_summary(signal),
        "leakage_audit": leakage.to_dict(orient="records") if not leakage.empty else [],
    }
    json_path = reports_dir / "error_robustness_report.json"
    md_path = reports_dir / "error_robustness_report.md"
    save_json(payload, json_path)
    md_path.write_text(_robustness_markdown(payload), encoding="utf-8")
    return [json_path, md_path]


def _log_report_metrics(reports_dir: Path) -> None:
    import mlflow

    for path, prefix in [
        (reports_dir / "validation_prediction_metrics.parquet", "validation"),
        (reports_dir / "test_prediction_metrics.parquet", "test"),
        (reports_dir / "test_signal_metrics.parquet", "test_signal"),
    ]:
        frame = _load_optional_table(path)
        if frame.empty:
            continue
        for key, value in _numeric_summary(frame).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                mlflow.log_metric(f"{prefix}_{key}", float(value))


def _numeric_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    summary: dict[str, float] = {}
    for col in frame.select_dtypes(include=[np.number]).columns:
        values = frame[col].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        summary[f"{col}_mean"] = float(values.mean())
        summary[f"{col}_min"] = float(values.min())
        summary[f"{col}_max"] = float(values.max())
    return summary


def _load_optional_table(path: Path) -> pd.DataFrame:
    try:
        return load_table(path)
    except FileNotFoundError:
        return pd.DataFrame()


def _robustness_markdown(payload: dict[str, Any]) -> str:
    selected = payload["selected_models"]
    lines = [
        f"# Error And Robustness Report: {payload['horizon_name']}",
        "",
        "## Selected Models",
        "",
    ]
    if selected:
        lines.append(pd.DataFrame(selected).to_string(index=False))
    else:
        lines.append("No selected models were found.")
    lines.extend(["", "## Numeric Summaries", ""])
    for key in ["validation_summary", "test_summary", "signal_summary"]:
        lines.append(f"### {key}")
        lines.append("")
        summary = payload.get(key, {})
        lines.append(pd.Series(summary).to_string() if summary else "No data.")
        lines.append("")
    return "\n".join(lines)


def _artifact_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    import hashlib

    hashes = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(root))] = digest
    return hashes


def _dependency_snapshot() -> dict[str, str]:
    return {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }


def _git_commit() -> str | None:
    return _run_git(["git", "rev-parse", "HEAD"])


def _git_status_short() -> str | None:
    return _run_git(["git", "status", "--short"])


def _run_git(cmd: list[str]) -> str | None:
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _set_mlflow_tracking_uri(tracking_uri: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)


def _json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in record.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        out[str(key)] = value
    return out


def _model_pip_requirements() -> list[str]:
    return [
        "mlflow==3.13.0",
        "joblib",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "pyyaml",
        "catboost",
        "lightgbm",
        "xgboost",
        "torch>=2.0",
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, log and register strict stock forecast models in MLflow.")
    parser.add_argument("--artifact-root", default="artifacts", help="Forecasting artifact root directory.")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI.")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--horizons", nargs="+", default=["week", "month"], choices=sorted(HORIZON_SPECS))
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--include-lstm", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
