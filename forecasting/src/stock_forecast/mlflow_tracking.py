from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"
MLFLOW_PARAM_MAX_LENGTH = 500
MLFLOW_TABLE_MAX_ROWS = 5000
MLFLOW_PREDICTION_SAMPLE_ROWS = 1000


@dataclass
class MLflowRunConfig:
    tracking_uri: str | None
    experiment_name: str
    notebook_name: str
    horizon_name: str
    horizon: int
    enabled: bool = True
    log_optuna_trials: bool = True
    run_name: str | None = None


@dataclass
class MLflowRunHandle:
    run_id: str | None
    trial_logger: MLflowTrialLogger | None


class MLflowTrialLogger:
    """Log Optuna trials as child runs without relying on process-local active runs."""

    def __init__(
        self,
        experiment_id: str,
        parent_run_id: str,
        notebook_name: str,
        horizon_name: str,
        horizon: int,
        enabled: bool = True,
    ):
        self.experiment_id = str(experiment_id)
        self.parent_run_id = str(parent_run_id)
        self.notebook_name = str(notebook_name)
        self.horizon_name = str(horizon_name)
        self.horizon = int(horizon)
        self.enabled = bool(enabled)

    def log_trial(
        self,
        *,
        trial_number: int,
        model_name: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        tags: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        run_name = f"{self.horizon_name}-{model_name}-trial-{int(trial_number)}"
        run_tags = _mlflow_tags(
            {
                MLFLOW_PARENT_RUN_ID: self.parent_run_id,
                "notebook_name": self.notebook_name,
                "horizon_name": self.horizon_name,
                "horizon": self.horizon,
                "model_name": model_name,
                "trial_number": int(trial_number),
                **dict(tags or {}),
            }
        )
        run = client.create_run(
            experiment_id=self.experiment_id,
            tags=run_tags,
            run_name=run_name,
        )
        status = "FINISHED"
        try:
            for key, value in _mlflow_params({"trial_number": int(trial_number), **params}).items():
                client.log_param(run.info.run_id, key, value)
            for key, value in _mlflow_metrics(metrics).items():
                client.log_metric(run.info.run_id, key, value)
        except Exception:
            status = "FAILED"
            raise
        finally:
            client.set_terminated(run.info.run_id, status=status)


@contextmanager
def mlflow_horizon_run(
    config: MLflowRunConfig,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
) -> Iterator[MLflowRunHandle]:
    """Start a parent MLflow run for a notebook/horizon training pass."""
    if not config.enabled:
        yield MLflowRunHandle(run_id=None, trial_logger=None)
        return

    import mlflow

    if config.tracking_uri:
        _allow_file_store_if_needed(config.tracking_uri)
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    run_name = config.run_name or f"{config.notebook_name}-{config.horizon_name}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            _mlflow_tags(
                {
                    "notebook_name": config.notebook_name,
                    "horizon_name": config.horizon_name,
                    "horizon": config.horizon,
                    "training_protocol": "strict",
                    **dict(tags or {}),
                }
            )
        )
        mlflow.log_params(
            _mlflow_params(
                {
                    "notebook_name": config.notebook_name,
                    "horizon_name": config.horizon_name,
                    "horizon": config.horizon,
                    **dict(params or {}),
                }
            )
        )
        experiment_id = run.info.experiment_id
        trial_logger = MLflowTrialLogger(
            experiment_id=experiment_id,
            parent_run_id=run.info.run_id,
            notebook_name=config.notebook_name,
            horizon_name=config.horizon_name,
            horizon=config.horizon,
            enabled=config.log_optuna_trials,
        )
        yield MLflowRunHandle(run_id=run.info.run_id, trial_logger=trial_logger)


def log_strict_protocol_result(
    result: dict[str, Any],
    *,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
) -> None:
    """Log final strict-protocol result tables, metrics, reports and model artifacts."""
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None:
        return

    if tags:
        mlflow.set_tags(_mlflow_tags(tags))
    if params:
        mlflow.log_params(_mlflow_params(params))

    for key, frame in _result_tables(result).items():
        if frame.empty:
            continue
        max_rows = MLFLOW_PREDICTION_SAMPLE_ROWS if "prediction" in key and "metrics" not in key else MLFLOW_TABLE_MAX_ROWS
        logged = frame.head(max_rows).copy()
        mlflow.log_table(_mlflow_table(logged), artifact_file=f"tables/{key}.json")
        mlflow.log_metric(f"{key}_logged_rows", float(len(logged)))
        mlflow.log_metric(f"{key}_total_rows", float(len(frame)))

    for prefix, key in [
        ("validation", "validation_prediction_metrics"),
        ("test", "test_prediction_metrics"),
        ("validation_signal", "validation_signal_metrics"),
        ("validation_panel", "validation_panel_signal_metrics"),
        ("test_signal", "test_signal_metrics"),
        ("test_panel", "test_panel_signal_metrics"),
    ]:
        frame = _as_frame(result.get(key))
        for metric_key, value in _numeric_summary(frame).items():
            mlflow.log_metric(_safe_mlflow_name(f"{prefix}_{metric_key}"), value)

    reports_dir = result.get("reports_dir")
    if reports_dir is not None and Path(reports_dir).exists():
        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
    models_dir = result.get("models_dir")
    if models_dir is not None and Path(models_dir).exists():
        mlflow.log_artifacts(str(models_dir), artifact_path="models")


def _result_tables(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    keys = [
        "outer_splits",
        "validation_model_ranking",
        "selected_models_by_ticker",
        "selected_models_by_model",
        "validation_prediction_metrics",
        "test_prediction_metrics",
        "validation_threshold_search",
        "validation_signal_metrics",
        "validation_panel_signal_metrics",
        "test_signal_metrics",
        "test_panel_signal_metrics",
        "inner_tuning_metrics",
        "best_params",
        "leakage_audit",
        "validation_predictions",
        "test_predictions",
    ]
    return {key: _as_frame(result.get(key)) for key in keys}


def _allow_file_store_if_needed(tracking_uri: str) -> None:
    if tracking_uri.startswith("file://") or "://" not in tracking_uri:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def _as_frame(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _numeric_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    out: dict[str, float] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        out[f"{column}_mean"] = float(values.mean())
        out[f"{column}_min"] = float(values.min())
        out[f"{column}_max"] = float(values.max())
    return out


def _mlflow_params(params: dict[str, Any]) -> dict[str, str]:
    out = {}
    for key, value in params.items():
        if value is None:
            continue
        out[_safe_mlflow_name(str(key))] = _stringify_mlflow_value(value)
    return out


def _mlflow_tags(tags: dict[str, Any]) -> dict[str, str]:
    return {str(key): _stringify_mlflow_value(value) for key, value in tags.items() if value is not None}


def _mlflow_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if _is_finite_number(value):
            out[_safe_mlflow_name(str(key))] = float(value)
    return out


def _mlflow_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].map(lambda value: value.isoformat() if pd.notna(value) else None)
        else:
            out[column] = out[column].map(_json_table_value)
    return out


def _safe_mlflow_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "value"


def _stringify_mlflow_value(value: Any) -> str:
    value = _json_safe_value(value)
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    return text[:MLFLOW_PARAM_MAX_LENGTH]


def _json_safe_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        pass
    elif pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _json_table_value(value: Any) -> Any:
    value = _json_safe_value(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if hasattr(value, "item"):
        value = value.item()
    return isinstance(value, (int, float)) and math.isfinite(float(value))
