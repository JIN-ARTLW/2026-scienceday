"""검증용 가상 위성 궤도 이력 (정답 BC와 정답 밀도 편향을 알고 있는 자료).

'실제 대기'를 MSIS × bias(태양활동, 지자기, 고도)로 두고 궤도를 적분한 뒤,
TLE처럼 불규칙한 시각·잡음·이상치·결측을 넣어 Orbitoby orbit_elements 형식으로 내보낸다.
파이프라인이 정답 BC를 복원하는지, 보정 모델이 편향을 학습하는지 확인하는 데 쓴다.
실제 연구 결과로 쓰면 안 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymsis import msis

from orbital_decay.constants import MU_EARTH, R_EARTH_WGS72_KM, R_EQ_KM
from orbital_decay.density import _sample_geometry
from orbital_decay.orbit import make_satrec
from orbital_decay.physics import drag_factor
from orbital_decay.spaceweather import daily_space_weather

J2 = 1.08262668e-3


def default_bias(f107_trail81, ap_trail3, alt_km):
    """MSIS 대비 '실제' 밀도 배율 (가상 정답). 활동이 강할수록, 고도가 높을수록 MSIS가 과소평가."""
    return (
        (f107_trail81 / 120.0) ** 0.3
        * (1 + 0.15 * np.tanh((ap_trail3 - 15.0) / 10.0))
        * (alt_km / 500.0) ** 0.4
    )


def simulate_satellite(
    norad_id: int,
    start: str,
    n_days: int,
    alt0_km: float,
    inc_deg: float,
    bc_true: float,
    space_weather: pd.DataFrame,
    rng: np.random.Generator,
    bias=default_bias,
    samples_per_day: int = 12,
    tle_noise_km: float = 0.05,
    outlier_rate: float = 0.01,
    gap_rate: float = 0.05,
):
    """반환: (orbit_elements 형식 DataFrame, 일별 정답 궤도 DataFrame)"""
    sw = space_weather.set_index("date")
    t0 = pd.Timestamp(start).floor("D")
    raan, argp, m0 = rng.uniform(0, 360, 3)
    ecc = 0.001
    a = alt0_km + R_EQ_KM
    truth = []
    step = pd.to_timedelta((np.arange(samples_per_day) + 0.5) / samples_per_day, unit="D")
    for d in range(n_days):
        day = t0 + pd.Timedelta(days=d)
        n_rev_day = np.sqrt(MU_EARTH / (a * 1e3) ** 3) * 86400 / (2 * np.pi)
        elem = pd.Series(
            {
                "epoch": day,
                "mean_motion": n_rev_day,
                "eccentricity": ecc,
                "inclination_deg": inc_deg,
                "raan_deg": raan,
                "arg_pericenter_deg": argp,
                "mean_anomaly_deg": m0,
                "bstar": 0.0,
            }
        )
        sat = make_satrec(elem)
        lon, lat, dalt = _sample_geometry(sat, day + step)
        alt = a - R_EQ_KM
        rho = msis.calculate((day + step).to_numpy().astype("datetime64[s]"), lon, lat, alt + dalt)[
            :, 0
        ].mean()
        env = sw.loc[day]
        rate = bc_true * bias(env["f107_trail81"], env["ap_trail3"], alt) * drag_factor(rho, a)
        truth.append(
            {
                "norad_id": norad_id,
                "time": day,
                "a_km": a,
                "alt_km": alt,
                "rate_km_day": rate,
                "mean_motion": n_rev_day,
                "raan_deg": raan,
            }
        )
        a += rate
        n_rad_s = np.sqrt(MU_EARTH / (a * 1e3) ** 3)
        raan_rate = -1.5 * n_rad_s * J2 * (R_EARTH_WGS72_KM / a) ** 2 * np.cos(np.radians(inc_deg))
        raan = (raan + np.degrees(raan_rate) * 86400) % 360
        if a - R_EQ_KM < 150:
            break
    truth = pd.DataFrame(truth)

    # TLE 발행: 대략 하루 1회, 시각 흔들림, 결측, 잡음, 이상치
    rows = []
    for i, r in truth.iloc[:-1].iterrows():
        if rng.random() < gap_rate:
            continue
        frac = rng.uniform(0, 1)
        nxt = truth.iloc[i + 1]
        a_t = r["a_km"] + (nxt["a_km"] - r["a_km"]) * frac
        a_t += rng.normal(0, tle_noise_km)
        if rng.random() < outlier_rate:
            a_t += rng.choice([-1, 1]) * rng.uniform(2, 5)
        rows.append(
            {
                "gp_id": norad_id * 100000 + i,
                "norad_id": norad_id,
                "epoch": r["time"] + pd.Timedelta(days=frac),
                "mean_motion": np.sqrt(MU_EARTH / (a_t * 1e3) ** 3) * 86400 / (2 * np.pi),
                "eccentricity": ecc,
                "inclination_deg": inc_deg,
                "raan_deg": r["raan_deg"],
                "arg_pericenter_deg": argp,
                "mean_anomaly_deg": rng.uniform(0, 360),
                "bstar": 1e-4,
                "source": "synthetic",
            }
        )
    return pd.DataFrame(rows), truth


def synthetic_fleet(
    n_satellites: int = 9,
    start: str = "2009-06-01",
    n_days: int = 1500,
    seed: int = 0,
    **kwargs,
):
    """고도·경사각·BC가 다른 가상 위성 여러 기. 반환: (elements, truth, 위성 제원 표)"""
    rng = np.random.default_rng(seed)
    sw = daily_space_weather(start, pd.Timestamp(start) + pd.Timedelta(days=n_days + 1))
    elements, truths, meta = [], [], []
    for k in range(n_satellites):
        norad_id = 90001 + k
        alt0 = rng.uniform(380, 560)
        inc = rng.choice([51.6, 65.0, 82.0, 97.5])
        bc = float(np.exp(rng.uniform(np.log(0.004), np.log(0.03))))
        el, tr = simulate_satellite(norad_id, start, n_days, alt0, inc, bc, sw, rng, **kwargs)
        elements.append(el)
        truths.append(tr)
        meta.append({"norad_id": norad_id, "alt0_km": alt0, "inc_deg": inc, "bc_true": bc})
    return (
        pd.concat(elements, ignore_index=True),
        pd.concat(truths, ignore_index=True),
        pd.DataFrame(meta),
    )
