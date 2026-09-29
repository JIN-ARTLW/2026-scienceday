"""가상 위성 감쇠 시뮬레이션 (뷰어의 시뮬레이션 탭, 위성 설계 질문용).

입력: 초기 고도, 경사각, BC(또는 질량·면적·C_D), 시작일, 기간, 태양활동 배율.
원궤도 가정. 속도를 위해
  1. 시뮬레이션 전체 기간 × 고도층(격자)의 궤도 평균 밀도를 MSIS 한 번 호출로 계산하고,
  2. ML 보정값도 같은 격자에서 한 번에 예측해 둔 뒤,
  3. 궤도 전파(physics.propagate)는 격자 보간만 한다.

태양·지자기 자료가 끝난 뒤(미래) 날짜는 태양 극대기를 맞춘 과거 주기(SC24)의 값을 다시 쓴다.
이는 예보가 아니라 '비슷한 주기가 반복된다면'이라는 시나리오다.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymsis import msis

from orbital_decay.constants import R_EQ_KM
from orbital_decay.density import _sample_geometry
from orbital_decay.orbit import make_satrec
from orbital_decay.physics import REENTRY_KM, drag_factor, propagate
from orbital_decay.spaceweather import SW_FEATURES_SCALED, daily_space_weather

# SC25 극대(2024-10 무렵)와 SC24 극대(2014-04 무렵)의 차이
ANALOG_OFFSET = pd.Timedelta(days=3835)
CD_DEFAULT = 2.2


@functools.cache
def space_weather_end() -> pd.Timestamp:
    """pymsis가 가진 태양·지자기 자료의 마지막 날짜.

    범위 밖 날짜로 조회하면 pymsis가 자료를 매번 새로 내려받으므로(수 초),
    이미 불러온 자료에서 읽는다. pymsis 내부 구조가 바뀌면 예외 메시지에서 읽는다.
    """
    msis.get_f107_ap(np.array(["2000-01-01"], dtype="datetime64[s]"))  # 자료 로드
    try:
        from pymsis.utils import _SPACE_WEATHER

        return pd.Timestamp(_SPACE_WEATHER.data["dates"][-1]).floor("D")
    except Exception:
        try:
            msis.get_f107_ap(np.array(["2300-01-01"], dtype="datetime64[s]"))
        except ValueError as exc:
            return pd.Timestamp(str(exc).rsplit("and", 1)[-1].strip(" .")).floor("D")
        return pd.Timestamp("2300-01-01")


def _source_times(times: pd.DatetimeIndex, end: pd.Timestamp) -> pd.DatetimeIndex:
    """자료가 없는 미래 시각은 과거 주기의 같은 위상 시각으로 옮긴다."""
    shift = np.where(times > end - pd.Timedelta(days=2), ANALOG_OFFSET, pd.Timedelta(0))
    return times - pd.to_timedelta(shift)


@dataclass
class Scenario:
    alt0_km: float = 450.0
    inc_deg: float = 51.6
    bc: float = 0.01  # C_D·A/m [m^2/kg]
    start: str = "2014-01-01"
    days: int = 1095
    f107_scale: float = 1.0
    area_to_mass: float | None = None  # [m^2/kg], 모델이 이 특징을 쓰면 필요

    @staticmethod
    def bc_from(mass_kg: float, area_m2: float, cd: float = CD_DEFAULT) -> float:
        return cd * area_m2 / mass_kg


class GridDensity:
    """날짜 × 평균고도 격자의 궤도 평균 밀도. DensityLookup과 같은 호출 방식."""

    def __init__(self, days: pd.DatetimeIndex, levels_km: np.ndarray, log_rho: np.ndarray):
        self.days = days.to_numpy().astype("datetime64[D]")
        self.levels = np.asarray(levels_km, dtype=float)
        self.log_rho = log_rho

    def dates(self, norad_id=None) -> np.ndarray:
        return self.days

    def _index(self, dates, alt_km):
        idx = np.clip(
            np.searchsorted(self.days, np.asarray(dates, dtype="datetime64[D]")),
            0,
            len(self.days) - 1,
        )
        step = self.levels[1] - self.levels[0]
        pos = (np.asarray(alt_km, dtype=float) - self.levels[0]) / step
        k = np.clip(np.floor(pos).astype(int), 0, len(self.levels) - 2)
        return idx, k, pos - k

    def rho(self, norad_id, dates, alt_km) -> np.ndarray:
        idx, k, f = self._index(dates, alt_km)
        y0, y1 = self.log_rho[idx, k], self.log_rho[idx, k + 1]
        return np.exp(y0 + (y1 - y0) * f)


def density_grid(
    sc: Scenario,
    step_km: float = 20.0,
    samples_per_day: int = 8,
    min_alt_km: float = 100.0,
) -> tuple[GridDensity, pd.DatetimeIndex]:
    """시나리오 전체 기간의 궤도 평균 밀도 격자. 반환: (격자, 날짜 목록)"""
    days = pd.date_range(pd.Timestamp(sc.start).floor("D"), periods=sc.days + 2, freq="D")
    levels = np.arange(min_alt_km, sc.alt0_km + 2 * step_km, step_km)

    n_rev_day = np.sqrt(3.986004418e14 / ((sc.alt0_km + R_EQ_KM) * 1e3) ** 3) * 86400 / (2 * np.pi)
    elem = pd.Series(
        {
            "epoch": days[0],
            "mean_motion": n_rev_day,
            "eccentricity": 0.0005,
            "inclination_deg": sc.inc_deg,
            "raan_deg": 0.0,
            "arg_pericenter_deg": 0.0,
            "mean_anomaly_deg": 0.0,
            "bstar": 0.0,
        }
    )
    sat = make_satrec(elem)  # 항력 없는 SGP4: 경사각·J2 세차가 반영된 궤도 형상만 쓴다
    S, L = samples_per_day, len(levels)
    step = pd.to_timedelta((np.arange(S) + 0.5) / S, unit="D")
    times = pd.DatetimeIndex((days.to_numpy()[:, None] + step.to_numpy()[None, :]).ravel())
    lon, lat, dalt = _sample_geometry(sat, times)

    end = space_weather_end()
    src = _source_times(times, end)
    f107, f107a, ap = msis.get_f107_ap(src.to_numpy().astype("datetime64[s]"))
    f107, f107a = f107 * sc.f107_scale, f107a * sc.f107_scale

    shape = (len(times), L)
    out = msis.calculate(
        np.broadcast_to(times.to_numpy().astype("datetime64[s]")[:, None], shape).ravel(),
        np.broadcast_to(lon[:, None], shape).ravel(),
        np.broadcast_to(lat[:, None], shape).ravel(),
        np.clip(levels[None, :] + dalt[:, None], 80.0, None).ravel(),
        f107s=np.broadcast_to(f107[:, None], shape).ravel(),
        f107as=np.broadcast_to(f107a[:, None], shape).ravel(),
        aps=np.broadcast_to(ap[:, None, :], (*shape, 7)).reshape(-1, 7),
    )
    rho = out[:, 0].reshape(len(days), S, L)
    return GridDensity(days, levels, np.log(np.nanmean(rho, axis=1))), days


def scenario_space_weather(sc: Scenario, days: pd.DatetimeIndex) -> pd.DataFrame:
    """ML 특징용 일별 지수 (미래는 과거 주기로 대체, F10.7 계열은 배율 적용)."""
    end = space_weather_end()
    src = _source_times(days, end)
    sw = daily_space_weather(src.min(), src.max()).set_index("date")
    sw = sw.reindex(src.floor("D")).reset_index(drop=True)
    sw[SW_FEATURES_SCALED] = sw[SW_FEATURES_SCALED] * sc.f107_scale
    sw.insert(0, "date", days)
    return sw


def _correction_grid(model, sc: Scenario, grid: GridDensity, sw: pd.DataFrame) -> np.ndarray:
    """(날짜, 고도층)마다 모델 배율을 한 번에 예측한다."""
    D, L = grid.log_rho.shape
    rows = sw.loc[np.repeat(np.arange(D), L)].reset_index(drop=True)
    alt = np.tile(grid.levels, D)
    rows["alt_mid_km"] = alt
    rows["inc_deg"] = sc.inc_deg
    rows["ecc"] = 0.0005
    rows["log_bc"] = np.log(sc.bc)
    if "area_to_mass" in getattr(model, "features", []):
        if sc.area_to_mass is None:
            raise ValueError(
                f"{model.name} 모델은 면적/질량비가 필요합니다 (Scenario.area_to_mass)"
            )
        rows["area_to_mass"] = sc.area_to_mass
    rho = np.exp(grid.log_rho.ravel())
    rows["log_phys_rate"] = np.log(-sc.bc * drag_factor(rho, alt + R_EQ_KM))
    return np.asarray(model.correction(rows), dtype=float).reshape(D, L)


def simulate(sc: Scenario, models: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """시나리오를 모델별로 전파한다.

    models: {"physics": None, "fixed_k": 모델, "ml": 모델, ...}  (None = 보정 없음)
    반환: (곡선 표 [time, alt_km, model], 정보 dict)
    """
    models = models if models is not None else {"physics": None}
    grid, days = density_grid(sc)
    sw = None
    curves, lifetimes = [], {}
    for name, model in models.items():
        correction = None
        if model is not None:
            if sw is None:
                sw = scenario_space_weather(sc, days)
            cgrid = _correction_grid(model, sc, grid, sw)

            def correction(dates, alt_km, cgrid=cgrid):
                idx, k, f = grid._index(dates, alt_km)
                return cgrid[idx, k] + (cgrid[idx, k + 1] - cgrid[idx, k]) * f

        curve = propagate(
            0,
            sc.alt0_km + R_EQ_KM,
            days[0],
            sc.days,
            sc.bc,
            grid,
            correction=correction,
            steps_per_day=2,
        )
        curve["model"] = name
        curves.append(curve)
        down = curve[curve["alt_km"] <= REENTRY_KM]
        lifetimes[name] = (down["time"].iloc[0] - days[0]).days if not down.empty else None
    end = space_weather_end()
    curves = pd.concat(curves, ignore_index=True)
    info = {
        "lifetime_days": lifetimes,
        # 재진입 전까지 실제로 과거 주기 자료를 쓴 경우에만 표시
        "uses_analog_after": end if curves["time"].max() > end else None,
    }
    return curves, info
