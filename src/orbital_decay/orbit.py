"""TLE/GP 궤도 이력 정리: 반장축 계산, 이상치 제거, 감쇠율 창(window) 계산.

입력은 Orbitoby `orbit_elements` 테이블과 같은 열 이름을 쓰는 DataFrame이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sgp4.api import WGS72, Satrec

from orbital_decay.constants import R_EARTH_WGS72_KM, R_EQ_KM

ELEMENT_COLUMNS = [
    "norad_id",
    "epoch",
    "mean_motion",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "arg_pericenter_deg",
    "mean_anomaly_deg",
    "bstar",
]

_SGP4_EPOCH0 = pd.Timestamp("1949-12-31")


def make_satrec(row) -> Satrec:
    """평균 궤도요소 한 행으로 SGP4 위성 객체를 만든다."""
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        0,
        (pd.Timestamp(row.epoch) - _SGP4_EPOCH0).total_seconds() / 86400.0,
        float(row.bstar),
        0.0,
        0.0,
        float(row.eccentricity),
        np.radians(row.arg_pericenter_deg),
        np.radians(row.inclination_deg),
        np.radians(row.mean_anomaly_deg),
        float(row.mean_motion) * 2 * np.pi / 1440.0,  # rev/day -> rad/min (Kozai)
        np.radians(row.raan_deg),
    )
    return sat


def prepare_history(
    elements: pd.DataFrame,
    outlier_sigma: float = 6.0,
    min_outlier_km: float = 0.2,
) -> pd.DataFrame:
    """궤도요소를 정리하고 Brouwer 평균 반장축과 평균고도를 붙인다.

    - 같은 위성·같은 epoch 중복은 마지막 값만 남긴다.
    - 이웃 TLE의 이동중앙값에서 크게 벗어난 반장축은 이상치로 제거한다.
    반환 DataFrame의 `attrs["cleaning"]`에 위성별 제거 개수를 기록한다.
    """
    missing = [c for c in ELEMENT_COLUMNS if c not in elements.columns]
    if missing:
        raise ValueError(f"궤도요소 열이 없습니다: {missing}")

    df = elements[ELEMENT_COLUMNS].dropna().copy()
    df["epoch"] = pd.to_datetime(df["epoch"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["norad_id", "epoch"]).drop_duplicates(["norad_id", "epoch"], keep="last")

    # SGP4 초기화 결과의 a는 Brouwer 평균 반장축(지구반지름 단위)
    df["a_km"] = [make_satrec(r).a * R_EARTH_WGS72_KM for r in df.itertuples()]
    df["alt_km"] = df["a_km"] - R_EQ_KM

    kept, report = [], {}
    for norad_id, g in df.groupby("norad_id", sort=True):
        resid = g["a_km"] - g["a_km"].rolling(9, center=True, min_periods=3).median()
        mad = 1.4826 * np.nanmedian(np.abs(resid))
        limit = max(outlier_sigma * mad, min_outlier_km)
        bad = resid.abs() > limit
        report[int(norad_id)] = {"input": len(g), "outliers": int(bad.sum())}
        kept.append(g[~bad])

    out = pd.concat(kept, ignore_index=True)
    out.attrs["cleaning"] = report
    return out


def decay_windows(
    history: pd.DataFrame,
    window_days: float = 7.0,
    min_points: int = 3,
) -> pd.DataFrame:
    """고정 길이 창마다 반장축을 직선 적합해 관측 감쇠율(km/day)을 구한다.

    TLE 한 쌍의 차분은 잡음이 커서, 창 안의 여러 TLE에 직선을 맞춘 기울기를 쓴다.
    창 경계는 위성의 첫 TLE가 속한 UTC 날짜 0시부터 시작한다.
    """
    rows = []
    for norad_id, g in history.groupby("norad_id", sort=True):
        t0 = g["epoch"].min().floor("D")
        t = (g["epoch"] - t0).dt.total_seconds().to_numpy() / 86400.0
        a = g["a_km"].to_numpy()
        idx = np.floor(t / window_days).astype(int)
        for k in np.unique(idx):
            m = idx == k
            if m.sum() < min_points:
                continue
            tk, ak = t[m], a[m]
            if tk.max() - tk.min() < window_days / 2:
                continue
            slope, intercept = np.polyfit(tk, ak, 1)
            start = k * window_days
            mid = start + window_days / 2
            rows.append(
                {
                    "norad_id": int(norad_id),
                    "t_start": t0 + pd.Timedelta(days=start),
                    "t_end": t0 + pd.Timedelta(days=start + window_days),
                    "a_start_km": intercept + slope * start,
                    "a_mid_km": intercept + slope * mid,
                    "obs_rate_km_day": slope,
                    "n_tle": int(m.sum()),
                    "inc_deg": float(g["inclination_deg"].to_numpy()[m].mean()),
                    "ecc": float(g["eccentricity"].to_numpy()[m].mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["alt_mid_km"] = out["a_mid_km"] - R_EQ_KM
        out["t_mid"] = out["t_start"] + (out["t_end"] - out["t_start"]) / 2
    return out
