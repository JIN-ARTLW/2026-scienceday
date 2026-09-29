"""대기항력에 의한 궤도 감쇠 물리모델.

근원궤도에서 평균 반장축의 변화율:

    da/dt = -BC · ρ · sqrt(μ a),   BC = C_D · A / m  [m^2/kg]

ρ는 궤도 평균 밀도이다. 대기 공전 효과·형상·C_D 불확실성은 모두 '유효 BC'에 흡수된다.
BC를 뺀 부분을 드래그 인자 g(ρ, a) = -ρ sqrt(μ a)로 두면 da/dt = BC · g 로 BC에 선형이다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from orbital_decay.constants import MU_EARTH, R_EQ_KM, SECONDS_PER_DAY
from orbital_decay.density import DensityLookup


def drag_factor(rho, a_km):
    """BC=1 m^2/kg일 때의 da/dt [km/day]."""
    return -np.asarray(rho) * np.sqrt(MU_EARTH * np.asarray(a_km) * 1e3) * SECONDS_PER_DAY / 1e3


def window_drag_factor(windows: pd.DataFrame, lookup: DensityLookup) -> np.ndarray:
    """각 감쇠율 창 동안의 평균 드래그 인자.

    창 안의 하루하루에 대해, 창의 직선 적합에서 얻은 그날 평균고도의 밀도를 쓴다.
    """
    out = np.empty(len(windows))
    for i, w in enumerate(windows.itertuples()):
        days = pd.date_range(w.t_start, w.t_end - pd.Timedelta(seconds=1), freq="D")
        dt_days = ((days + pd.Timedelta(hours=12)) - w.t_mid).total_seconds() / 86400
        a = w.a_mid_km + w.obs_rate_km_day * dt_days.to_numpy()
        rho = lookup.rho(w.norad_id, days, a - R_EQ_KM)
        out[i] = drag_factor(rho, a).mean()
    return out


def estimate_bc(obs_rate: np.ndarray, factor: np.ndarray) -> float:
    """최소제곱 BC: obs ≈ BC · g  →  BC = Σ obs·g / Σ g²."""
    obs_rate, factor = np.asarray(obs_rate), np.asarray(factor)
    ok = np.isfinite(obs_rate) & np.isfinite(factor)
    return float(np.sum(obs_rate[ok] * factor[ok]) / np.sum(factor[ok] ** 2))


def propagate(
    norad_id: int,
    a0_km: float,
    t0: pd.Timestamp,
    n_days: int,
    bc: float,
    lookup: DensityLookup,
    correction: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    steps_per_day: int = 4,
) -> pd.DataFrame:
    """평균 반장축을 t0부터 n_days 동안 적분한다 (RK2, 밀도는 날짜별 표에서 보간).

    correction(dates, alt_km) -> 감쇠율 배율. 보정계수 모델과 ML 보정모델이 여기로 들어온다.
    """
    dt = 1.0 / steps_per_day
    t0 = pd.Timestamp(t0)
    available = lookup.dates(norad_id).astype("datetime64[D]")
    last_day = available.max()

    def rate(t_days: float, a_km: float) -> float:
        day = (t0 + pd.Timedelta(days=t_days)).to_datetime64().astype("datetime64[D]")
        day = min(day, last_day)
        alt = a_km - R_EQ_KM
        if alt < 120:
            return 0.0
        r = bc * drag_factor(lookup.rho(norad_id, [day], alt), a_km)[0]
        if correction is not None:
            r *= float(correction(np.array([day]), np.array([alt]))[0])
        return r

    t, a = 0.0, float(a0_km)
    rows = [(t0, a)]
    for _ in range(n_days):
        for _ in range(steps_per_day):
            k1 = rate(t, a)
            k2 = rate(t + dt / 2, a + k1 * dt / 2)
            a += k2 * dt
            t += dt
        rows.append((t0 + pd.Timedelta(days=t), a))
        if a - R_EQ_KM < 120:
            break
    out = pd.DataFrame(rows, columns=["time", "a_km"])
    out["alt_km"] = out["a_km"] - R_EQ_KM
    return out
