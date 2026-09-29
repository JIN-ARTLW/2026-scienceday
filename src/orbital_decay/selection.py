"""ML 보정모델 후보 비교와 근거 있는 선택 (위성 단위 중첩 교차검증).

모든 후보는 같은 조건에서 비교한다.
  - 같은 특징(models.DEFAULT_FEATURES), 같은 목표(배율 = 관측/물리), 같은 가중치(물리²)
    → 모두 감쇠율 제곱오차를 최소화하도록 학습된다.
  - 바깥 루프: 위성 하나씩 빼기(LOSO). 빠진 위성은 학습·튜닝 어디에도 쓰이지 않는다.
  - 안쪽 루프: 학습 위성만으로 GroupKFold(위성 단위) → 하이퍼파라미터 선택.
  - 위성별 점수: skill = 1 - RMSE_모델 / RMSE_순수물리 (위성마다 감쇠 크기가 달라서 비율로 비교).

선택 규칙 (select_model):
  1. 위성 평균 skill이 가장 높은 후보를 찾는다 (순수 물리모델, 고정 보정계수 모델도 후보).
  2. 그 후보와 위성별 RMSE를 짝지어 Wilcoxon 부호순위검정을 하고,
     유의하게 나쁘지 않은(p ≥ alpha) 후보 중 가장 단순한 것을 고른다 (과적합·설명 부담 최소화).
  3. 고른 모델이 고정 보정계수 모델보다 유의하게 나은지도 함께 보고한다.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from orbital_decay.models import (
    DEFAULT_FEATURES,
    MIN_CORRECTION,
    FixedCorrectionModel,
    PhysicsModel,
    _training_rows,
    _weights,
)

MAX_CORRECTION = 5.0


class WeightedKNN(RegressorMixin, BaseEstimator):
    """가중 k-최근접 이웃: 이웃 배율의 가중평균 (scikit-learn KNN은 sample_weight 미지원)."""

    def __init__(self, n_neighbors: int = 30):
        self.n_neighbors = n_neighbors

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        self.nn_ = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X))).fit(X)
        self.y_ = np.asarray(y, dtype=float)
        self.w_ = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight)
        return self

    def predict(self, X):
        _, idx = self.nn_.kneighbors(np.asarray(X, dtype=float))
        w = self.w_[idx]
        return (w * self.y_[idx]).sum(axis=1) / w.sum(axis=1)


class WeightedGP(RegressorMixin, BaseEstimator):
    """가우시안 과정 회귀. 표본 가중치는 표본별 잡음분산(noise / w)으로 반영한다.

    계산량이 n³이라 max_samples까지만 무작위로 뽑아 학습한다.
    """

    def __init__(self, noise: float = 0.1, max_samples: int = 1500, random_state: int = 0):
        self.noise = noise
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if len(X) > self.max_samples:
            keep = np.random.default_rng(self.random_state).choice(
                len(X), self.max_samples, replace=False
            )
            X, y, w = X[keep], y[keep], w[keep]
        kernel = ConstantKernel(1.0) * RBF(np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3))
        self.gp_ = GaussianProcessRegressor(
            kernel, alpha=self.noise / np.maximum(w / w.mean(), 1e-3), normalize_y=True
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.gp_.predict(np.asarray(X, dtype=float))


def _scaled(model) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", model)])


@dataclass
class Candidate:
    """비교 후보. complexity가 낮을수록 단순(동률일 때 우선)."""

    name: str
    family: str
    complexity: int
    build: object  # (params) -> estimator
    grid: list[dict]
    note: str = ""


def available_candidates(random_state: int = 0) -> list[Candidate]:
    rs = random_state
    cands = [
        Candidate(
            "ridge",
            "선형",
            2,
            lambda p: _scaled(Ridge(**p)),
            [{"alpha": a} for a in (0.1, 10.0, 1000.0)],
            "특징의 선형 결합. 해석이 가장 쉽다.",
        ),
        Candidate(
            "ridge_poly2",
            "선형+상호작용",
            3,
            lambda p: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("poly", PolynomialFeatures(2, include_bias=False)),
                    ("model", Ridge(**p)),
                ]
            ),
            [{"alpha": a} for a in (1.0, 100.0, 10000.0)],
            "2차항과 변수 간 곱(예: 고도×F10.7)까지 포함한 선형 모델.",
        ),
        Candidate(
            "knn",
            "이웃 기반",
            4,
            lambda p: _scaled(WeightedKNN(**p)),
            [{"n_neighbors": k} for k in (15, 50, 150)],
            "비슷한 조건의 과거 창들의 배율 평균.",
        ),
        Candidate(
            "svr",
            "커널",
            5,
            lambda p: _scaled(SVR(kernel="rbf", gamma="scale", **p)),
            [{"C": c, "epsilon": 0.05} for c in (0.3, 3.0)],
            "RBF 커널 서포트 벡터 회귀.",
        ),
        Candidate(
            "gaussian_process",
            "커널(베이즈)",
            5,
            lambda p: _scaled(WeightedGP(random_state=rs, **p)),
            [{"noise": n} for n in (0.05, 0.5)],
            "불확실성까지 주는 커널 회귀. 표본 1500개까지만 사용.",
        ),
        Candidate(
            "random_forest",
            "트리 앙상블(배깅)",
            6,
            lambda p: RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=1, **p),
            [
                {"max_depth": 6, "min_samples_leaf": 10},
                {"max_depth": None, "min_samples_leaf": 40},
                {"max_depth": 10, "min_samples_leaf": 20},
            ],
            "",
        ),
        Candidate(
            "extra_trees",
            "트리 앙상블(배깅)",
            6,
            lambda p: ExtraTreesRegressor(n_estimators=300, random_state=rs, n_jobs=1, **p),
            [
                {"max_depth": 6, "min_samples_leaf": 10},
                {"max_depth": None, "min_samples_leaf": 40},
                {"max_depth": 10, "min_samples_leaf": 20},
            ],
            "",
        ),
        Candidate(
            "hist_gbm",
            "트리 부스팅",
            7,
            lambda p: HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=rs, **p
            ),
            [
                {"max_depth": 3, "min_samples_leaf": 20},
                {"max_depth": 5, "min_samples_leaf": 20},
                {"max_depth": None, "min_samples_leaf": 40},
            ],
            "scikit-learn 부스팅 (LightGBM 방식).",
        ),
        Candidate(
            "mlp",
            "신경망",
            8,
            lambda p: _scaled(
                MLPRegressor(max_iter=800, early_stopping=True, random_state=rs, **p)
            ),
            [
                {"hidden_layer_sizes": h, "alpha": a}
                for h in ((32,), (64, 32))
                for a in (1e-3, 1e-1)
            ],
            "다층 퍼셉트론.",
        ),
    ]
    try:
        from xgboost import XGBRegressor

        XGBRegressor()  # libomp 없으면 여기서 실패
        cands.append(
            Candidate(
                "xgboost",
                "트리 부스팅",
                7,
                lambda p: XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    n_jobs=1,
                    random_state=rs,
                    **p,
                ),
                [{"max_depth": d, "min_child_weight": 5} for d in (3, 5)],
            )
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        cands.append(
            Candidate(
                "lightgbm",
                "트리 부스팅",
                7,
                lambda p: LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.03,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    n_jobs=1,
                    random_state=rs,
                    verbose=-1,
                    **p,
                ),
                [{"num_leaves": n, "min_child_samples": 20} for n in (15, 31)],
            )
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        cands.append(
            Candidate(
                "catboost",
                "트리 부스팅",
                7,
                lambda p: CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.05,
                    verbose=False,
                    thread_count=1,
                    random_seed=rs,
                    **p,
                ),
                [{"depth": d} for d in (4, 6)],
            )
        )
    except Exception:
        pass
    return cands


# 후보에서 뺀 방법과 이유 (보고서에 그대로 쓴다)
EXCLUDED = {
    "LSTM/GRU/Transformer (순환·시퀀스 신경망)": (
        "입력 단위가 '창 하나의 특징'이고, 창 사이 시간 의존은 후행 평균 특징으로 이미 표현했다. "
        "위성 수(수십 기 이하)에 비해 매개변수가 많아 과적합 위험이 크다. "
        "MLP가 트리/선형보다 뚜렷이 나을 때만 추가 실험 가치가 있다."
    ),
    "CNN": "이미지·격자 자료용. 표 형태 특징에는 구조적 이점이 없다.",
    "로지스틱 회귀·분류기": "예측 대상이 연속값(감쇠율 배율)이라 해당하지 않는다.",
    "물리정보신경망(PINN)": (
        "물리식(da/dt = -BC·ρ·√(μa))은 이미 기준모델로 들어 있고 ML은 그 잔차만 배운다. "
        "같은 목적을 더 단순하게 달성한다."
    ),
}


def _fit(estimator, rows: pd.DataFrame, features):
    estimator = clone(estimator)
    kw = "model__sample_weight" if isinstance(estimator, Pipeline) else "sample_weight"
    estimator.fit(rows[features], rows["ratio"], **{kw: _weights(rows)})
    return estimator


def _weighted_mse(estimator, rows, features) -> float:
    pred = np.clip(estimator.predict(rows[features]), MIN_CORRECTION, MAX_CORRECTION)
    w = rows["weight"].to_numpy()
    return float(np.sum(w * (rows["ratio"].to_numpy() - pred) ** 2) / np.sum(w))


class CandidateCorrection(PhysicsModel):
    """후보 하나를 물리모델 보정 배율로 쓰는 모델. tune=True면 안쪽 GroupKFold로 격자 선택."""

    def __init__(
        self, candidate: Candidate, features=None, tune: bool = True, inner_folds: int = 3
    ):
        self.candidate = candidate
        self.name = candidate.name
        self.features = features or DEFAULT_FEATURES
        self.tune = tune
        self.inner_folds = inner_folds
        self.max_correction = MAX_CORRECTION

    def fit(self, df: pd.DataFrame) -> CandidateCorrection:
        rows = _training_rows(df)
        grid = self.candidate.grid
        if self.tune and len(grid) > 1 and rows["norad_id"].nunique() >= 2:
            n_splits = min(self.inner_folds, rows["norad_id"].nunique())
            scores = []
            for params in grid:
                errs = []
                for tr, va in GroupKFold(n_splits).split(rows, groups=rows["norad_id"]):
                    est = _fit(self.candidate.build(params), rows.iloc[tr], self.features)
                    errs.append(_weighted_mse(est, rows.iloc[va], self.features))
                scores.append(np.mean(errs))
            self.params_ = grid[int(np.argmin(scores))]
        else:
            self.params_ = grid[0]
        self.model_ = _fit(self.candidate.build(self.params_), rows, self.features)
        return self

    def correction(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.model_.predict(df[self.features])
        return np.clip(pred, MIN_CORRECTION, MAX_CORRECTION)


def _outer_fold(df, held_out, models):
    train = df[df["norad_id"] != held_out]
    test = df[(df["norad_id"] == held_out) & ~df["is_calib"]]
    obs = test["obs_rate_km_day"].to_numpy() * 1e3
    rows = []
    for model in models:
        t = time.perf_counter()
        m = copy.deepcopy(model).fit(train)
        pred = m.predict_rate(test) * 1e3
        err = pred - obs
        rows.append(
            {
                "norad_id": int(held_out),
                "model": m.name,
                "n": len(test),
                "rmse_m_day": float(np.sqrt(np.mean(err**2))),
                "mae_m_day": float(np.mean(np.abs(err))),
                "mean_abs_obs_m_day": float(np.mean(np.abs(obs))),
                "params": str(getattr(m, "params_", "")),
                "fit_seconds": time.perf_counter() - t,
            }
        )
    return rows


def nested_loso(
    df: pd.DataFrame,
    candidates: list[Candidate] | None = None,
    features=None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """바깥 LOSO × 안쪽 GroupKFold. 반환: 위성×모델 오차 표 (skill 포함)."""
    candidates = candidates or available_candidates()
    models = [PhysicsModel(), FixedCorrectionModel()] + [
        CandidateCorrection(c, features) for c in candidates
    ]
    ids = sorted(df["norad_id"].unique())
    folds = Parallel(n_jobs=n_jobs)(delayed(_outer_fold)(df, i, models) for i in ids)
    res = pd.DataFrame([r for fold in folds for r in fold])
    phys = res[res["model"] == "physics"].set_index("norad_id")["rmse_m_day"]
    res["skill"] = 1 - res["rmse_m_day"] / res["norad_id"].map(phys)
    return res


def select_model(
    results: pd.DataFrame,
    candidates: list[Candidate] | None = None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
):
    """근거 표와 최종 선택. 반환: (요약 표, 선택된 모델 이름, 설명 문장 목록)"""
    candidates = candidates or available_candidates()
    complexity = {"physics": 0, "fixed_k": 1, **{c.name: c.complexity for c in candidates}}
    family = {
        "physics": "기준(물리)",
        "fixed_k": "기준(배율 1개)",
        **{c.name: c.family for c in candidates},
    }
    rmse = results.pivot(index="norad_id", columns="model", values="rmse_m_day")
    skill = results.pivot(index="norad_id", columns="model", values="skill")
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(skill), (n_boot, len(skill)))

    best = skill.mean().idxmax()
    rows = []
    for m in skill.columns:
        boots = skill[m].to_numpy()[boot_idx].mean(axis=1)
        p_vs_best = (
            1.0
            if m == best or np.allclose(rmse[m], rmse[best])
            else wilcoxon(rmse[m], rmse[best]).pvalue
        )
        p_vs_k = (
            np.nan
            if m == "fixed_k" or np.allclose(rmse[m], rmse["fixed_k"])
            else wilcoxon(rmse[m], rmse["fixed_k"], alternative="less").pvalue
        )
        rows.append(
            {
                "model": m,
                "family": family.get(m, ""),
                "complexity": complexity.get(m, 9),
                "mean_skill": skill[m].mean(),
                "skill_ci_low": np.percentile(boots, 2.5),
                "skill_ci_high": np.percentile(boots, 97.5),
                "median_skill": skill[m].median(),
                "sats_better_than_physics": int((skill[m] > 0).sum()),
                "sats_better_than_fixed_k": int((rmse[m] < rmse["fixed_k"]).sum()),
                "p_vs_best": p_vs_best,
                "p_better_than_fixed_k": p_vs_k,
                "fit_seconds": results.loc[results["model"] == m, "fit_seconds"].mean(),
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_skill", ascending=False).set_index("model")

    tied = table[table["p_vs_best"] >= alpha]
    chosen = tied.sort_values(["complexity", "mean_skill"], ascending=[True, False]).index[0]
    n = len(skill)
    c = table.loc[chosen]
    reasons = [
        f"위성 {n}기 LOSO 중첩 교차검증에서 평균 skill 최고는 {best} "
        f"({table.loc[best, 'mean_skill']:.3f}).",
        f"{best}와 위성별 RMSE가 유의하게 다르지 않은(Wilcoxon p ≥ {alpha}) 후보: "
        f"{', '.join(tied.index)}.",
        f"그중 가장 단순한 {chosen}({c['family']})을 선택: 평균 skill {c['mean_skill']:.3f} "
        f"[95% CI {c['skill_ci_low']:.3f}, {c['skill_ci_high']:.3f}], "
        f"물리모델보다 나은 위성 {c['sats_better_than_physics']}/{n}기.",
    ]
    if chosen == "physics":
        reasons.append(
            "어떤 보정 모델도 순수 물리모델보다 유의하게 낫지 않다 → "
            "보정 없이 물리모델을 쓰는 것이 근거에 맞다. "
            "(이 자체가 보고할 결과: 보정이 일반화되지 않는 이유를 위성별로 분석)"
        )
    elif chosen != "fixed_k":
        p = c["p_better_than_fixed_k"]
        reasons.append(
            f"고정 보정계수 모델 대비: {c['sats_better_than_fixed_k']}/{n}기에서 우세, "
            f"단측 Wilcoxon p = {p:.3g} → "
            + (
                "ML 보정이 유의하게 낫다."
                if p < alpha
                else "유의한 개선이라고 말할 근거는 아직 부족하다 "
                "(위성 수를 늘리거나 결과를 조심스럽게 해석)."
            )
        )
    if len(skill) < 10:
        reasons.append(f"주의: 위성이 {n}기뿐이라 검정력이 낮다. 10기 이상 권장.")
    return table, chosen, reasons
