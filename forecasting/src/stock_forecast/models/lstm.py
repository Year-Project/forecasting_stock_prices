from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .base import ModelSpec


FULL_FRAME_EXCLUDE_COLUMNS = {"date", "ticker", "target_date", "entry_date"}
try:
    import torch as _torch
    from torch import nn as _torch_nn
except ImportError:  # pragma: no cover - optional dependency guard
    _torch = None
    _torch_nn = None


@dataclass
class SequenceSamples:
    x: np.ndarray
    y: np.ndarray | None
    end_index: list[Any]


def build_lstm_sequence_samples(
    frame: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    target_col: str | None = None,
) -> SequenceSamples:
    """Build sequence samples independently within each ticker."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    required = {"date", "ticker", *feature_cols}
    if target_col is not None:
        required.add(target_col)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns for LSTM sequences: {missing}")

    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    end_index: list[Any] = []
    for _, group in frame.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        group = group.sort_values("date")
        features = group[feature_cols].to_numpy(dtype=float)
        targets = group[target_col].to_numpy(dtype=float) if target_col is not None else None
        indices = list(group.index)
        if len(group) < lookback:
            continue
        for end_pos in range(lookback - 1, len(group)):
            if targets is not None and not np.isfinite(targets[end_pos]):
                continue
            x_values.append(features[end_pos - lookback + 1 : end_pos + 1])
            if targets is not None:
                y_values.append(float(targets[end_pos]))
            end_index.append(indices[end_pos])

    if not x_values:
        n_features = len(feature_cols)
        return SequenceSamples(
            x=np.empty((0, lookback, n_features), dtype=np.float32),
            y=np.empty((0,), dtype=np.float32) if target_col is not None else None,
            end_index=[],
        )
    return SequenceSamples(
        x=np.asarray(x_values, dtype=np.float32),
        y=np.asarray(y_values, dtype=np.float32) if target_col is not None else None,
        end_index=end_index,
    )


if _torch_nn is not None:

    class ReturnLSTMNetwork(_torch_nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            input_projection_size: int,
            lstm_dropout: float,
            head_dropout: float,
        ):
            super().__init__()
            if input_projection_size > 0:
                self.projection = _torch_nn.Sequential(
                    _torch_nn.Linear(input_size, input_projection_size),
                    _torch_nn.LayerNorm(input_projection_size),
                    _torch_nn.SiLU(),
                    _torch_nn.Dropout(head_dropout),
                )
                lstm_input_size = input_projection_size
            else:
                self.projection = None
                lstm_input_size = input_size
            self.lstm = _torch_nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = _torch_nn.Sequential(
                _torch_nn.LayerNorm(hidden_size),
                _torch_nn.Dropout(head_dropout),
                _torch_nn.Linear(hidden_size, 1),
            )

        def forward(self, x):
            if self.projection is not None:
                x = self.projection(x)
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)

else:

    class ReturnLSTMNetwork:  # pragma: no cover - torch import path raises before construction
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError("PyTorch is required for the LSTM model. Install the forecasting[deep] extra.")


class LSTMReturnRegressor(BaseEstimator, RegressorMixin):
    """PyTorch sequence-to-one LSTM regressor for future return targets."""

    def __init__(
        self,
        lookback: int = 60,
        hidden_size: int = 64,
        num_layers: int = 1,
        input_projection_size: int = 0,
        lstm_dropout: float = 0.0,
        head_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        loss: str = "smooth_l1",
        huber_beta: float = 0.05,
        grad_clip_norm: float = 1.0,
        max_epochs: int = 150,
        patience: int = 15,
        validation_fraction: float = 0.2,
        feature_clip: float = 5.0,
        device: str = "auto",
        random_state: int = 42,
        ensemble_seeds: list[int] | tuple[int, ...] | None = None,
        verbose: bool = False,
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_projection_size = input_projection_size
        self.lstm_dropout = lstm_dropout
        self.head_dropout = head_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.loss = loss
        self.huber_beta = huber_beta
        self.grad_clip_norm = grad_clip_norm
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.feature_clip = feature_clip
        self.device = device
        self.random_state = random_state
        self.ensemble_seeds = ensemble_seeds
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "LSTMReturnRegressor":
        torch, nn, DataLoader, TensorDataset = _torch_modules()
        self._validate_params()
        data = self._prepare_fit_frame(X, y)
        self.feature_cols_ = _infer_feature_columns(data)
        self.fallback_ = _finite_mean(data["_target"].to_numpy(dtype=float))
        self.training_context_ = data[["date", "ticker", *self.feature_cols_]].copy()

        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = RobustScaler()
        feature_values = data[self.feature_cols_].to_numpy(dtype=float)
        self.imputer_.fit(feature_values)
        self.scaler_.fit(self.imputer_.transform(feature_values))

        samples = build_lstm_sequence_samples(
            self._transform_frame(data),
            feature_cols=self.feature_cols_,
            lookback=int(self.lookback),
            target_col="_target",
        )
        self.n_sequences_ = int(len(samples.x))
        if samples.y is None or len(samples.y) == 0:
            self.model_ = None
            self.training_history_ = {"fallback_only": True, "epochs_ran": 0, "best_epoch": 0}
            return self

        y_values = samples.y.astype(np.float32)
        self.target_mean_ = float(np.mean(y_values))
        target_std = float(np.std(y_values))
        self.target_scale_ = target_std if target_std > 1e-8 else 1.0
        y_scaled = ((y_values - self.target_mean_) / self.target_scale_).astype(np.float32)

        train_idx, valid_idx = _chronological_sequence_split(len(y_scaled), float(self.validation_fraction))
        x_train = samples.x[train_idx]
        y_train = y_scaled[train_idx]
        x_valid = samples.x[valid_idx] if len(valid_idx) else samples.x[train_idx]
        y_valid = y_scaled[valid_idx] if len(valid_idx) else y_scaled[train_idx]

        device = _resolve_device(str(self.device), torch)
        ensemble_seeds = _ensemble_seed_values(self.ensemble_seeds, int(self.random_state))
        self.models_ = []
        member_histories = []
        for seed in ensemble_seeds:
            network, history = _fit_lstm_member(
                seed=seed,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                input_size=len(self.feature_cols_),
                hidden_size=int(self.hidden_size),
                num_layers=int(self.num_layers),
                input_projection_size=int(self.input_projection_size),
                lstm_dropout=float(self.lstm_dropout),
                head_dropout=float(self.head_dropout),
                loss=str(self.loss),
                huber_beta=float(self.huber_beta),
                learning_rate=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
                batch_size=int(self.batch_size),
                grad_clip_norm=float(self.grad_clip_norm),
                max_epochs=int(self.max_epochs),
                patience=int(self.patience),
                device=device,
                torch=torch,
                nn=nn,
                DataLoader=DataLoader,
                TensorDataset=TensorDataset,
                verbose=bool(self.verbose),
            )
            self.models_.append(network)
            member_histories.append(history)
        self.model_ = self.models_[0] if self.models_ else None
        first_history = member_histories[0] if member_histories else {}
        self.training_history_ = {
            "fallback_only": False,
            "ensemble_size": len(self.models_),
            "ensemble_seeds": ensemble_seeds,
            "epochs_ran": max([int(history["epochs_ran"]) for history in member_histories] or [0]),
            "best_epoch": first_history.get("best_epoch", 0),
            "best_validation_loss": min([float(history["best_validation_loss"]) for history in member_histories] or [float("inf")]),
            "train_loss": first_history.get("train_loss", []),
            "validation_loss": first_history.get("validation_loss", []),
            "member_histories": member_histories,
            "device": str(device),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError("LSTMReturnRegressor must be fitted before prediction")
        if self.model_ is None:
            return np.full(len(X), float(self.fallback_), dtype=float)
        torch, _, DataLoader, TensorDataset = _torch_modules()
        pred_frame = X.copy()
        pred_frame["date"] = pd.to_datetime(pred_frame["date"])
        pred_frame["_prediction_order"] = np.arange(len(pred_frame))
        pred_frame["_is_prediction"] = True

        context = self.training_context_.copy()
        context["_prediction_order"] = -1
        context["_is_prediction"] = False
        combined = pd.concat([context, pred_frame[context.columns]], ignore_index=True)
        combined = combined.sort_values(["ticker", "date", "_is_prediction"]).drop_duplicates(
            ["ticker", "date"],
            keep="last",
        )
        combined = self._transform_frame(combined)

        sequences, orders = self._prediction_sequences(combined)
        predictions = np.full(len(pred_frame), float(self.fallback_), dtype=float)
        if len(sequences) == 0:
            return predictions

        loader = DataLoader(
            TensorDataset(torch.as_tensor(sequences.astype(np.float32))),
            batch_size=int(self.batch_size),
            shuffle=False,
        )
        member_predictions = []
        models = list(getattr(self, "models_", [])) or [self.model_]
        with torch.no_grad():
            for model in models:
                values = []
                model.eval()
                for (x_batch,) in loader:
                    pred = model(x_batch).detach().cpu().numpy().astype(float)
                    values.append(pred)
                member_predictions.append(np.concatenate(values) if values else np.empty((0,), dtype=float))
        scaled_pred = np.mean(np.vstack(member_predictions), axis=0) if member_predictions else np.empty((0,), dtype=float)
        return_pred = scaled_pred * float(self.target_scale_) + float(self.target_mean_)
        predictions[np.asarray(orders, dtype=int)] = return_pred
        return np.nan_to_num(predictions, nan=float(self.fallback_), posinf=float(self.fallback_), neginf=float(self.fallback_))

    def _prepare_fit_frame(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        if not {"date", "ticker"}.issubset(X.columns):
            raise ValueError("LSTMReturnRegressor requires date and ticker columns")
        out = X.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["_target"] = np.asarray(y, dtype=float)
        out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        values = out[self.feature_cols_].to_numpy(dtype=float)
        transformed = self.scaler_.transform(self.imputer_.transform(values))
        if float(self.feature_clip) > 0:
            transformed = np.clip(transformed, -float(self.feature_clip), float(self.feature_clip))
        out.loc[:, self.feature_cols_] = transformed
        return out

    def _prediction_sequences(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
        x_values = []
        orders = []
        for _, group in frame.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
            group = group.sort_values("date")
            features = group[self.feature_cols_].to_numpy(dtype=np.float32)
            is_prediction = group["_is_prediction"].to_numpy(dtype=bool)
            order = group["_prediction_order"].to_numpy(dtype=int)
            if len(group) < int(self.lookback):
                continue
            for end_pos in range(int(self.lookback) - 1, len(group)):
                if not is_prediction[end_pos]:
                    continue
                seq = features[end_pos - int(self.lookback) + 1 : end_pos + 1]
                if np.isfinite(seq).all():
                    x_values.append(seq)
                    orders.append(int(order[end_pos]))
        if not x_values:
            return np.empty((0, int(self.lookback), len(self.feature_cols_)), dtype=np.float32), []
        return np.asarray(x_values, dtype=np.float32), orders

    def _validate_params(self) -> None:
        if int(self.lookback) <= 0:
            raise ValueError("lookback must be positive")
        if int(self.hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if int(self.num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if str(self.loss) not in {"smooth_l1", "mse"}:
            raise ValueError("loss must be one of: smooth_l1, mse")
        if float(self.huber_beta) <= 0:
            raise ValueError("huber_beta must be positive")
        if float(self.feature_clip) < 0:
            raise ValueError("feature_clip must be non-negative")
        _ensemble_seed_values(self.ensemble_seeds, int(self.random_state))


def build_lstm_model(
    random_state: int = 42,
    params: dict[str, object] | None = None,
) -> ModelSpec:
    defaults: dict[str, object] = {"random_state": random_state}
    defaults.update(params or {})
    return ModelSpec(
        "lstm",
        LSTMReturnRegressor(**defaults),
        scale_features=False,
        use_pipeline=False,
        input_mode="full_frame",
    )


def _infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in FULL_FRAME_EXCLUDE_COLUMNS
        and not col.startswith("_")
        and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else 0.0


def _chronological_sequence_split(n_rows: int, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if n_rows <= 2 or not 0.0 < validation_fraction < 1.0:
        idx = np.arange(n_rows)
        return idx, np.array([], dtype=int)
    valid_size = max(1, int(round(n_rows * validation_fraction)))
    if n_rows - valid_size < 1:
        valid_size = 1
    train_end = n_rows - valid_size
    return np.arange(train_end), np.arange(train_end, n_rows)


def _torch_modules():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("PyTorch is required for the LSTM model. Install the forecasting[deep] extra.") from exc
    return torch, nn, DataLoader, TensorDataset


def _resolve_device(device: str, torch: Any):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_network(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    input_projection_size: int,
    lstm_dropout: float,
    head_dropout: float,
    nn: Any,
):
    return ReturnLSTMNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        input_projection_size=input_projection_size,
        lstm_dropout=lstm_dropout,
        head_dropout=head_dropout,
    )


def _loss_function(loss: str, huber_beta: float, nn: Any):
    if loss == "mse":
        return nn.MSELoss()
    return nn.SmoothL1Loss(beta=huber_beta)


def _ensemble_seed_values(ensemble_seeds: list[int] | tuple[int, ...] | None, random_state: int) -> list[int]:
    if ensemble_seeds is None:
        return [int(random_state)]
    seeds = [int(seed) for seed in ensemble_seeds]
    return seeds or [int(random_state)]


def _fit_lstm_member(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    input_projection_size: int,
    lstm_dropout: float,
    head_dropout: float,
    loss: str,
    huber_beta: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    grad_clip_norm: float,
    max_epochs: int,
    patience: int,
    device: Any,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    verbose: bool,
) -> tuple[Any, dict[str, Any]]:
    _set_torch_seed(seed, torch)
    network = _build_network(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        input_projection_size=input_projection_size,
        lstm_dropout=lstm_dropout,
        head_dropout=head_dropout,
        nn=nn,
    ).to(device)
    criterion = _loss_function(loss, huber_beta, nn)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, patience // 3),
    )
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_valid), torch.as_tensor(y_valid)),
        batch_size=batch_size,
        shuffle=False,
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    train_losses = []
    valid_losses = []
    for epoch in range(1, max_epochs + 1):
        network.train()
        train_loss = _run_train_epoch(network, train_loader, criterion, optimizer, device, grad_clip_norm, nn)
        valid_loss = _evaluate_loss(network, valid_loader, criterion, device)
        scheduler.step(valid_loss)
        train_losses.append(float(train_loss))
        valid_losses.append(float(valid_loss))
        if valid_loss < best_loss - 1e-8:
            best_loss = float(valid_loss)
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"seed={seed} epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")
        if no_improve >= patience:
            break

    if best_state is not None:
        network.load_state_dict(best_state)
    network = network.to("cpu")
    network.eval()
    return network, {
        "seed": int(seed),
        "epochs_ran": len(train_losses),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train_loss": train_losses,
        "validation_loss": valid_losses,
    }


def _set_torch_seed(seed: int, torch: Any) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


def _run_train_epoch(network: Any, loader: Any, criterion: Any, optimizer: Any, device: Any, grad_clip_norm: float, nn: Any) -> float:
    network.train()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = network(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(y_batch)
        total_count += len(y_batch)
    return total_loss / max(total_count, 1)


def _evaluate_loss(network: Any, loader: Any, criterion: Any, device: Any) -> float:
    network.eval()
    total_loss = 0.0
    total_count = 0
    import torch

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = network(x_batch)
            loss = criterion(pred, y_batch)
            total_loss += float(loss.detach().cpu()) * len(y_batch)
            total_count += len(y_batch)
    return total_loss / max(total_count, 1)
