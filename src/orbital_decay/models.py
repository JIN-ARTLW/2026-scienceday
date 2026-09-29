"""비교할 세 감쇠 예측 모델.

1. PhysicsModel          순수 물리모델: BC(보정 구간 추정) × MSIS 드래그 인자
2. FixedCorrectionModel  물리모델 × 학습 위성들에서 얻은 하나의 보정계수 k
3. MLCorrectionModel     물리모델 × f(특징), f는 학습 위성의 배율(관측/물리)을 학습한 부스팅 트리

k와 f 모두 감쇠율 제곱오차 Σ(관측 - 배율·물리)²를 최소화하도록 학습한다
(배율 목표 + 가중치 물리²; dataset.build_dataset 참고).

모두 '감쇠율 배율(correction)'로 표현되므로 같은 궤도 전파기(physics.propagate)에 넣을 수 있다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from orbital_decay.dataset import SW_FEATURES

# 위성 ID 대신 새 위성에서도 구할 수 있는 물리·환경 특징만 쓴다.
DEFAULT_FEATURES = [
    "alt_mid_km",
    "inc_deg",
    "ecc",
    "log_bc",
    "log_phys_rate",
    *SW_FEATURES,
]


MIN_CORRECTION = 0.05


def _training_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["is_calib"] & np.isfinite(df["ratio"])]


def _weights(rows: pd.DataFrame) -> np.ndarray:
    w = rows["weight"].to_numpy()
    return w / w.mean()


class PhysicsModel:
    name = "physics"

    def fit(self, df: pd.DataFrame) -> PhysicsModel:
        return self

    def correction(self, df: pd.DataFrame) -> np.ndarray:
        return np.ones(len(df))

    def predict_rate(self, df: pd.DataFrame) -> np.ndarray:
        return df["phys_rate_km_day"].to_numpy() * self.correction(df)


class FixedCorrectionModel(PhysicsModel):
    """k = Σ 관측·물리 / Σ 물리² (학습 위성 평가 구간의 최소제곱 배율)."""

    name = "fixed_k"

    def fit(self, df: pd.DataFrame) -> FixedCorrectionModel:
        rows = _training_rows(df)
        self.k_ = float(
            np.sum(rows["obs_rate_km_day"] * rows["phys_rate_km_day"]) / np.sum(rows["weight"])
        )
        return self

    def correction(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.k_)


def _make_regressor(backend: str, random_state: int):
    if backend in ("auto", "xgboost"):
        try:
            from xgboost import XGBRegressor

            return "xgboost", XGBRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=1.0,
                random_state=random_state,
            )
        except Exception:
            if backend == "xgboost":
                raise
    from sklearn.ensemble import HistGradientBoostingRegressor

    return "sklearn_hgb", HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=random_state,
    )


class MLCorrectionModel(PhysicsModel):
    """물리 예측에 곱할 배율을 부스팅 트리로 학습한다 (가중 제곱오차).

    backend: "auto"(XGBoost가 되면 XGBoost, 아니면 scikit-learn HistGradientBoosting),
             "xgboost", "sklearn".
    """

    name = "ml"

    def __init__(
        self,
        features: list[str] | None = None,
        backend: str = "auto",
        random_state: int = 0,
    ):
        self.features = features or DEFAULT_FEATURES
        self.backend = backend
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> MLCorrectionModel:
        rows = _training_rows(df)
        self.backend_, self.model_ = _make_regressor(self.backend, self.random_state)
        self.model_.fit(rows[self.features], rows["ratio"], sample_weight=_weights(rows))
        return self

    def correction(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model_.predict(df[self.features]), MIN_CORRECTION, None)

    def feature_importance(self, df: pd.DataFrame, n_repeats: int = 5) -> pd.Series:
        return feature_importance(self, df, n_repeats, self.random_state)


def feature_importance(
    model: PhysicsModel, df: pd.DataFrame, n_repeats: int = 5, random_state: int = 0
) -> pd.Series:
    """학습된 보정모델의 순열 중요도 (가중 배율 MSE 증가량). 보정 학습이 없는 모델은 빈 값."""
    if not hasattr(model, "model_"):
        return pd.Series(dtype=float)
    from sklearn.inspection import permutation_importance

    rows = _training_rows(df)
    result = permutation_importance(
        model.model_,
        rows[model.features],
        rows["ratio"],
        sample_weight=_weights(rows),
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )
    return pd.Series(result.importances_mean, index=model.features).sort_values(ascending=False)


class FittedCorrection(PhysicsModel):
    """학습이 끝난 보정모델을 저장·공유하기 위한 형태 (예측에 필요한 것만 담는다).

    후보 정의(Candidate)처럼 pickle할 수 없는 객체를 떼어내고, 뷰어·시뮬레이션에서
    correction()만 쓰도록 한다.
    """

    def __init__(self, name, model, features, min_correction=MIN_CORRECTION, max_correction=None):
        self.name = name
        self.model_ = model
        self.features = list(features)
        self.min_correction = min_correction
        self.max_correction = max_correction

    def fit(self, df: pd.DataFrame) -> FittedCorrection:
        return self

    def correction(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(
            self.model_.predict(df[self.features]), self.min_correction, self.max_correction
        )


def freeze(model: PhysicsModel) -> PhysicsModel:
    """학습된 모델을 저장 가능한 형태로 바꾼다 (물리·고정 k 모델은 그대로)."""
    if not hasattr(model, "model_"):
        return model
    return FittedCorrection(
        model.name,
        model.model_,
        model.features,
        max_correction=getattr(model, "max_correction", None),
    )
