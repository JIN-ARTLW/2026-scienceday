"""감쇠율 창 + 물리모델 + 우주환경 특징을 합친 분석용 표.

각 위성은 앞쪽 calib_days 동안의 창(보정 구간)으로만 BC를 추정한다.
그 뒤의 창(평가 구간)이 모델 비교와 ML 학습·검증에 쓰인다.
학습에 쓰지 않은 위성도 자기 보정 구간만 있으면 똑같이 BC를 얻으므로,
'처음 보는 위성' 평가에 미래 정보가 섞이지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from orbital_decay.density import DensityLookup
from orbital_decay.physics import estimate_bc, window_drag_factor

SW_FEATURES = ["f107_obs", "f107_trail27", "f107_trail81", "ap_daily", "ap_trail3", "ap_max3"]


def build_dataset(
    windows: pd.DataFrame,
    lookup: DensityLookup,
    space_weather: pd.DataFrame,
    calib_days: float = 365.0,
    min_calib_windows: int = 8,
    metadata: pd.DataFrame | None = None,
    min_alt_km: float = 250.0,
) -> pd.DataFrame:
    """창 표에 BC, 물리 예측, 특징, 학습 목표를 붙인다.

    metadata(선택): norad_id, mass_kg, area_m2 열이 있으면 면적질량비 A/m을 특징으로 추가.
    min_alt_km: 재진입 직전(감쇠가 창 안에서 급가속해 직선 적합이 깨지는 구간)의 창을 제외한다.

    학습 목표는 ratio(가중치 weight)이고, log_ratio = ln(관측/물리)는 분석·그림용이다
    (관측 감쇠율이 0 이상인 창은 NaN).
    """
    low = windows["alt_mid_km"] < min_alt_km
    df = windows[~low].copy()
    df["drag_factor"] = window_drag_factor(df, lookup)

    first = df.groupby("norad_id")["t_start"].transform("min")
    df["days_since_start"] = (df["t_mid"] - first).dt.total_seconds() / 86400
    df["is_calib"] = df["days_since_start"] < calib_days

    bc = {}
    for norad_id, g in df[df["is_calib"]].groupby("norad_id"):
        if len(g) >= min_calib_windows:
            bc[norad_id] = estimate_bc(g["obs_rate_km_day"], g["drag_factor"])
    df["bc_est"] = df["norad_id"].map(bc)
    dropped = sorted(set(df["norad_id"]) - {k for k, v in bc.items() if v > 0})
    df = df[df["bc_est"] > 0].copy()

    df["phys_rate_km_day"] = df["bc_est"] * df["drag_factor"]
    # 학습 목표: 배율 ratio = 관측/물리, 가중치 phys² → 가중 제곱오차 = (관측 - 배율·물리)²
    # 즉 감쇠율 자체의 제곱오차를 최소화한다 (잡음으로 관측이 0 근처일 때도 안정).
    df["ratio"] = df["obs_rate_km_day"] / df["phys_rate_km_day"]
    df["weight"] = df["phys_rate_km_day"] ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        df["log_ratio"] = np.log(df["obs_rate_km_day"] / df["phys_rate_km_day"])
    df.loc[df["obs_rate_km_day"] >= 0, "log_ratio"] = np.nan

    # 창 중앙 날짜의 우주환경 (후행 평균만 사용 → 미래 정보 없음)
    sw = space_weather.set_index("date")[SW_FEATURES]
    df = df.join(sw, on=df["t_mid"].dt.floor("D").rename("date"))
    df["log_bc"] = np.log(df["bc_est"])
    df["log_phys_rate"] = np.log(-df["phys_rate_km_day"])

    if metadata is not None and {"mass_kg", "area_m2"} <= set(metadata.columns):
        am = metadata.set_index("norad_id").eval("area_m2 / mass_kg").rename("area_to_mass")
        df = df.join(am, on="norad_id")

    df = df.reset_index(drop=True)
    df.attrs["dropped_no_bc"] = dropped
    df.attrs["excluded_low_alt_windows"] = int(low.sum())
    return df
