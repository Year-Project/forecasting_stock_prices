from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .base import ModelSpec
from .lstm import (
    FULL_FRAME_EXCLUDE_COLUMNS,
    SequenceSamples,
    _balanced_ticker_indices,
    _chronological_sequence_split_by_date,
    _ensemble_seed_values,
    _evaluate_loss,
    _finite_mean,
    _infer_feature_columns,
    _inverse_scale_targets,
    _loss_function,
    _resolve_device,
    _run_train_epoch,
    _scale_targets,
    _set_torch_seed,
    _target_normalization_stats,
    build_lstm_sequence_samples,
)

try:
    import torch as _torch
    from torch import nn as _torch_nn
except ImportError:  # pragma: no cover - optional dependency guard
    _torch = None
    _torch_nn = None


def build_mamba_sequence_samples(
    frame: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    target_col: str | None = None,
) -> SequenceSamples:
    """Build sequence samples independently within each ticker."""
    return build_lstm_sequence_samples(frame, feature_cols, lookback, target_col)


if _torch_nn is not None:

    class ResidualMambaBlock(_torch_nn.Module):
        def __init__(
            self,
            model_dim: int,
            d_state: int,
            d_conv: int,
            expand: int,
            block_dropout: float,
            mamba_cls: Any,
        ):
            super().__init__()
            self.norm = _torch_nn.LayerNorm(model_dim)
            self.mamba = mamba_cls(
                d_model=model_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.dropout = _torch_nn.Dropout(block_dropout)

        def forward(self, x):
            return x + self.dropout(self.mamba(self.norm(x)))


    class ReturnMambaNetwork(_torch_nn.Module):
        def __init__(
            self,
            input_size: int,
            model_dim: int,
            num_layers: int,
            d_state: int,
            d_conv: int,
            expand: int,
            block_dropout: float,
            head_dropout: float,
            mamba_cls: Any,
        ):
            super().__init__()
            self.input_projection = _torch_nn.Sequential(
                _torch_nn.Linear(input_size, model_dim),
                _torch_nn.LayerNorm(model_dim),
                _torch_nn.SiLU(),
                _torch_nn.Dropout(head_dropout),
            )
            self.blocks = _torch_nn.ModuleList(
                [
                    ResidualMambaBlock(
                        model_dim=model_dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        block_dropout=block_dropout,
                        mamba_cls=mamba_cls,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.head = _torch_nn.Sequential(
                _torch_nn.LayerNorm(model_dim),
                _torch_nn.Dropout(head_dropout),
                _torch_nn.Linear(model_dim, 1),
            )

        def forward(self, x):
            x = self.input_projection(x)
            for block in self.blocks:
                x = block(x)
            return self.head(x[:, -1, :]).squeeze(-1)

else:

    class ReturnMambaNetwork:  # pragma: no cover - dependency import path raises first
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(_mamba_install_message())


@dataclass
class MambaTrainingConfig:
    lookback: int
    model_dim: int
    num_layers: int
    d_state: int
    d_conv: int
    expand: int
    block_dropout: float
    head_dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    loss: str
    huber_beta: float
    grad_clip_norm: float
    max_epochs: int
    patience: int


class MambaReturnRegressor(BaseEstimator, RegressorMixin):
    """Official mamba-ssm sequence-to-one regressor for future return targets."""

    def __init__(
        self,
        lookback: int = 126,
        model_dim: int = 64,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        block_dropout: float = 0.1,
        head_dropout: float = 0.1,
        learning_rate: float = 8e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        loss: str = "smooth_l1",
        huber_beta: float = 0.05,
        grad_clip_norm: float = 1.0,
        max_epochs: int = 120,
        patience: int = 12,
        validation_fraction: float = 0.2,
        feature_clip: float = 5.0,
        target_normalization: str = "per_ticker",
        balanced_ticker_sampling: bool = True,
        device: str = "auto",
        random_state: int = 42,
        ensemble_seeds: list[int] | tuple[int, ...] | None = None,
        verbose: bool = False,
    ):
        self.lookback = lookback
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.block_dropout = block_dropout
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
        self.target_normalization = target_normalization
        self.balanced_ticker_sampling = balanced_ticker_sampling
        self.device = device
        self.random_state = random_state
        self.ensemble_seeds = ensemble_seeds
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "MambaReturnRegressor":
        torch, nn, DataLoader, TensorDataset, _ = _mamba_modules()
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

        samples = build_mamba_sequence_samples(
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
        sample_tickers = np.asarray(samples.ticker, dtype=object)
        self.target_stats_ = _target_normalization_stats(
            y_values,
            sample_tickers,
            str(self.target_normalization),
        )
        self.target_mean_ = float(self.target_stats_["__global__"]["mean"])
        self.target_scale_ = float(self.target_stats_["__global__"]["scale"])
        y_scaled = _scale_targets(
            y_values,
            sample_tickers,
            self.target_stats_,
            str(self.target_normalization),
        )

        train_idx, valid_idx = _chronological_sequence_split_by_date(
            samples.end_date,
            float(self.validation_fraction),
        )
        balanced_info = None
        if bool(self.balanced_ticker_sampling):
            train_idx, balanced_info = _balanced_ticker_indices(
                train_idx,
                sample_tickers,
                random_state=int(self.random_state),
            )
        x_train = samples.x[train_idx]
        y_train = y_scaled[train_idx]
        x_valid = samples.x[valid_idx] if len(valid_idx) else samples.x[train_idx]
        y_valid = y_scaled[valid_idx] if len(valid_idx) else y_scaled[train_idx]

        device = _resolve_device(str(self.device), torch)
        ensemble_seeds = _ensemble_seed_values(self.ensemble_seeds, int(self.random_state))
        config = MambaTrainingConfig(
            lookback=int(self.lookback),
            model_dim=int(self.model_dim),
            num_layers=int(self.num_layers),
            d_state=int(self.d_state),
            d_conv=int(self.d_conv),
            expand=int(self.expand),
            block_dropout=float(self.block_dropout),
            head_dropout=float(self.head_dropout),
            learning_rate=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
            batch_size=int(self.batch_size),
            loss=str(self.loss),
            huber_beta=float(self.huber_beta),
            grad_clip_norm=float(self.grad_clip_norm),
            max_epochs=int(self.max_epochs),
            patience=int(self.patience),
        )
        self.models_ = []
        member_histories = []
        for seed in ensemble_seeds:
            network, history = _fit_mamba_member(
                seed=seed,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                input_size=len(self.feature_cols_),
                config=config,
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
            "target_normalization": str(self.target_normalization),
            "balanced_ticker_sampling": bool(self.balanced_ticker_sampling),
            "balanced_ticker_sampling_info": balanced_info,
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError("MambaReturnRegressor must be fitted before prediction")
        if self.model_ is None:
            return np.full(len(X), float(self.fallback_), dtype=float)
        torch, _, DataLoader, TensorDataset, _ = _mamba_modules()
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

        sequences, orders, tickers = self._prediction_sequences(combined)
        predictions = np.full(len(pred_frame), float(self.fallback_), dtype=float)
        if len(sequences) == 0:
            return predictions

        loader = DataLoader(
            TensorDataset(torch.as_tensor(sequences.astype(np.float32))),
            batch_size=int(self.batch_size),
            shuffle=False,
        )
        device = _resolve_device(str(self.device), torch)
        member_predictions = []
        models = list(getattr(self, "models_", [])) or [self.model_]
        with torch.no_grad():
            for model in models:
                values = []
                model = model.to(device)
                model.eval()
                for (x_batch,) in loader:
                    pred = model(x_batch.to(device)).detach().cpu().numpy().astype(float)
                    values.append(pred)
                member_predictions.append(np.concatenate(values) if values else np.empty((0,), dtype=float))
                model.to("cpu")
        scaled_pred = np.mean(np.vstack(member_predictions), axis=0) if member_predictions else np.empty((0,), dtype=float)
        return_pred = _inverse_scale_targets(
            scaled_pred,
            np.asarray(tickers, dtype=object),
            self.target_stats_,
            str(self.target_normalization),
        )
        predictions[np.asarray(orders, dtype=int)] = return_pred
        return np.nan_to_num(predictions, nan=float(self.fallback_), posinf=float(self.fallback_), neginf=float(self.fallback_))

    def _prepare_fit_frame(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        if not {"date", "ticker"}.issubset(X.columns):
            raise ValueError("MambaReturnRegressor requires date and ticker columns")
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

    def _prediction_sequences(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[int], list[str]]:
        x_values = []
        orders = []
        tickers = []
        for ticker, group in frame.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
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
                    tickers.append(str(ticker))
        if not x_values:
            return np.empty((0, int(self.lookback), len(self.feature_cols_)), dtype=np.float32), [], []
        return np.asarray(x_values, dtype=np.float32), orders, tickers

    def _validate_params(self) -> None:
        if int(self.lookback) <= 0:
            raise ValueError("lookback must be positive")
        if int(self.model_dim) <= 0:
            raise ValueError("model_dim must be positive")
        if int(self.num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if int(self.d_state) <= 0:
            raise ValueError("d_state must be positive")
        if not 2 <= int(self.d_conv) <= 4:
            raise ValueError("d_conv must be between 2 and 4 for the official causal_conv1d kernel")
        if int(self.expand) <= 0:
            raise ValueError("expand must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if str(self.loss) not in {"smooth_l1", "huber", "mse"}:
            raise ValueError("loss must be one of: smooth_l1, huber, mse")
        if float(self.huber_beta) <= 0:
            raise ValueError("huber_beta must be positive")
        if float(self.feature_clip) < 0:
            raise ValueError("feature_clip must be non-negative")
        if str(self.target_normalization) not in {"global", "per_ticker"}:
            raise ValueError("target_normalization must be one of: global, per_ticker")
        _ensemble_seed_values(self.ensemble_seeds, int(self.random_state))


def build_mamba_model(
    random_state: int = 42,
    params: dict[str, object] | None = None,
) -> ModelSpec:
    defaults: dict[str, object] = {"random_state": random_state}
    defaults.update(params or {})
    return ModelSpec(
        "mamba",
        MambaReturnRegressor(**defaults),
        scale_features=False,
        use_pipeline=False,
        input_mode="full_frame",
    )


def _mamba_modules():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(_mamba_install_message()) from exc
    try:
        os.environ.setdefault("TILELANG_CACHE_DIR", "/tmp/tilelang")
        from mamba_ssm import Mamba
    except Exception as exc:
        raise ImportError(_mamba_install_message()) from exc
    return torch, nn, DataLoader, TensorDataset, Mamba


def _build_network(
    input_size: int,
    model_dim: int,
    num_layers: int,
    d_state: int,
    d_conv: int,
    expand: int,
    block_dropout: float,
    head_dropout: float,
    nn: Any,
    Mamba: Any,
):
    if _torch_nn is not None:
        return ReturnMambaNetwork(
            input_size=input_size,
            model_dim=model_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            block_dropout=block_dropout,
            head_dropout=head_dropout,
            mamba_cls=Mamba,
        )
    raise ImportError(_mamba_install_message())


def _fit_mamba_member(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    input_size: int,
    config: MambaTrainingConfig,
    device: Any,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    verbose: bool,
) -> tuple[Any, dict[str, Any]]:
    _, _, _, _, Mamba = _mamba_modules()
    _set_torch_seed(seed, torch)
    network = _build_network(
        input_size=input_size,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        block_dropout=config.block_dropout,
        head_dropout=config.head_dropout,
        nn=nn,
        Mamba=Mamba,
    ).to(device)
    criterion = _loss_function(config.loss, config.huber_beta, nn)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, config.patience // 3),
    )
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_valid), torch.as_tensor(y_valid)),
        batch_size=config.batch_size,
        shuffle=False,
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    train_losses = []
    valid_losses = []
    for epoch in range(1, config.max_epochs + 1):
        network.train()
        train_loss = _run_train_epoch(network, train_loader, criterion, optimizer, device, config.grad_clip_norm, nn)
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
        if no_improve >= config.patience:
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


def _mamba_install_message() -> str:
    return (
        "Official mamba-ssm is required for the Mamba model. Install PyTorch first, then install "
        "`mamba-ssm[causal-conv1d]` with `--no-build-isolation` in the deep-learning environment."
    )
