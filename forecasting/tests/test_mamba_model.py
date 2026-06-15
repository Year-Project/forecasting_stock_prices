import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_forecast.mamba_features import get_mamba_feature_columns, make_mamba_features
from stock_forecast.models import build_model
from stock_forecast.models.mamba import MambaReturnRegressor


def _mamba_ssm_importable() -> bool:
    if importlib.util.find_spec("mamba_ssm") is None:
        return False
    os.environ.setdefault("TILELANG_CACHE_DIR", "/tmp/tilelang")
    try:
        import mamba_ssm  # noqa: F401
    except Exception:
        return False
    return True


def _ohlcv_frame(periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = 100.0 + np.arange(periods) * 0.08 + np.sin(np.arange(periods) / 6.0)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "open": close - 0.2,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": 1000 + np.arange(periods) * 4,
        }
    )


def test_mamba_features_do_not_depend_on_future_rows():
    base = _ohlcv_frame()
    changed_future = base.copy()
    changed_future.loc[70:, ["open", "high", "low", "close", "volume"]] = [300.0, 320.0, 280.0, 310.0, 999999.0]

    before = make_mamba_features(base)
    after = make_mamba_features(changed_future)
    feature_cols = get_mamba_feature_columns(before)

    assert_frame_equal(
        before.loc[:69, feature_cols],
        after.loc[:69, feature_cols],
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_mamba_factory_builds_full_frame_estimator_without_runtime_dependency():
    spec = build_model("mamba", random_state=7, params={"lookback": 4, "model_dim": 8, "num_layers": 1})

    assert spec.name == "mamba"
    assert spec.input_mode == "full_frame"
    assert spec.use_pipeline is False
    assert isinstance(spec.estimator, MambaReturnRegressor)

    if not _mamba_ssm_importable():
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "ticker": "A",
                "target_date": pd.date_range("2024-01-06", periods=10, freq="D"),
                "f1": np.linspace(0.0, 1.0, 10),
            }
        )
        with pytest.raises(ImportError, match="mamba-ssm"):
            spec.estimator.fit(frame, np.linspace(0.0, 0.1, 10))


def test_mamba_rejects_invalid_causal_conv_width_before_training():
    model = MambaReturnRegressor(d_conv=5)

    with pytest.raises(ValueError, match="d_conv must be between 2 and 4"):
        model._validate_params()


def test_mamba_regressor_fits_and_predicts_when_official_package_is_available():
    if not _mamba_ssm_importable():
        pytest.skip("official mamba-ssm is not importable in this environment")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("official mamba-ssm training is CUDA-oriented")

    periods = 32
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "target_date": dates + pd.Timedelta(days=5),
            "f1": np.linspace(0.0, 1.0, periods),
            "f2": np.cos(np.arange(periods) / 4.0),
        }
    )
    target = 0.01 * np.sin(np.arange(periods) / 5.0)

    model = MambaReturnRegressor(
        lookback=4,
        model_dim=8,
        num_layers=1,
        d_state=4,
        d_conv=2,
        expand=1,
        max_epochs=1,
        patience=1,
        batch_size=8,
        random_state=7,
        device="cuda",
    )
    model.fit(frame.iloc[:24].copy(), target[:24])
    pred = model.predict(frame.iloc[24:].copy())

    assert len(pred) == len(frame.iloc[24:])
    assert np.isfinite(pred).all()
