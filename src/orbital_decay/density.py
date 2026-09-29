"""궤도 평균 대기밀도 표 (NRLMSIS 2.1).

위성별·날짜별로 실제 궤도 경로(위도·경도·지방시)를 따라 MSIS 밀도를 표본 추출해 평균한다.
관측 고도 주변 여러 고도층에서 계산해 두므로, 예측 궤도가 관측과 달라져도(궤도 전파)
같은 표에서 밀도를 보간해 쓸 수 있다.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pymsis import msis

from orbital_decay.constants import R_EARTH_WGS72_KM, R_EQ_KM, WGS84_F
from orbital_decay.orbit import make_satrec

# 관측 평균고도 기준 고도층 (km)
OFFSETS_KM = np.array([-100.0, -50.0, -20.0, 0.0, 20.0, 50.0, 100.0])
LEVEL_COLUMNS = [f"log_rho_{i}" for i in range(len(OFFSETS_KM))]

_UNIX_JD = 2440587.5


def _local_radius_km(geodetic_lat_rad: np.ndarray) -> np.ndarray:
    b = R_EQ_KM * (1 - WGS84_F)
    c, s = np.cos(geodetic_lat_rad), np.sin(geodetic_lat_rad)
    return np.sqrt(((R_EQ_KM**2 * c) ** 2 + (b**2 * s) ** 2) / ((R_EQ_KM * c) ** 2 + (b * s) ** 2))


def _sample_geometry(sat, times: pd.DatetimeIndex):
    """SGP4로 표본 시각의 위도·경도와 '평균고도 대비 실제 고도 차이'를 구한다."""
    jd_full = times.to_numpy().astype("datetime64[ns]").astype(np.int64) / 86400e9 + _UNIX_JD
    jd = np.floor(jd_full)
    fr = jd_full - jd
    err, r, _ = sat.sgp4_array(jd, fr)
    r = np.where(err[:, None] == 0, r, np.nan)

    gmst = np.radians((280.46061837 + 360.98564736629 * (jd_full - 2451545.0)) % 360.0)
    lon = np.degrees(np.arctan2(r[:, 1], r[:, 0]) - gmst)
    lon = (lon + 180.0) % 360.0 - 180.0
    rho_xy = np.hypot(r[:, 0], r[:, 1])
    geocentric = np.arctan2(r[:, 2], rho_xy)
    geodetic = np.arctan(np.tan(geocentric) / (1 - WGS84_F) ** 2)
    alt = np.linalg.norm(r, axis=1) - _local_radius_km(geodetic)
    mean_alt = sat.a * R_EARTH_WGS72_KM - R_EQ_KM
    return lon, np.degrees(geodetic), alt - mean_alt


def satellite_density_table(
    history: pd.DataFrame,
    samples_per_day: int = 24,
    end=None,
    end_pad_days: int = 45,
) -> pd.DataFrame:
    """위성 하나의 날짜별 궤도 평균 밀도 표.

    기본 범위는 첫 TLE 날짜 ~ 마지막 TLE 날짜 + end_pad_days (마지막 감쇠율 창이 끝까지 덮이도록).

    열: norad_id, date, alt_center_km(그날 관측 평균고도), log_rho_0..6
    (log_rho_i = OFFSETS_KM[i]만큼 떨어진 평균고도에서의 궤도 평균 ln ρ[kg/m^3])
    """
    g = history.sort_values("epoch")
    norad_id = int(g["norad_id"].iloc[0])
    first = g["epoch"].min().floor("D")
    last = (
        pd.Timestamp(end).floor("D")
        if end is not None
        else g["epoch"].max().floor("D") + pd.Timedelta(days=end_pad_days)
    )
    days = pd.date_range(first, last, freq="D")

    # 형상 계산용 SGP4는 항력 없이(bstar=0) 초기화해 하루 안에서 고도가 흐르지 않게 한다
    sats = [make_satrec(r._replace(bstar=0.0)) for r in g.itertuples()]
    epochs = g["epoch"].to_numpy()
    noon = (days + pd.Timedelta(hours=12)).to_numpy()
    which = np.clip(np.searchsorted(epochs, noon, side="right") - 1, 0, len(sats) - 1)

    t_epoch = (g["epoch"] - first).dt.total_seconds().to_numpy() / 86400
    t_day = np.arange(len(days)) + 0.5
    center = np.interp(t_day, t_epoch, g["alt_km"].to_numpy())

    S, L = samples_per_day, len(OFFSETS_KM)
    step = pd.to_timedelta((np.arange(S) + 0.5) / S, unit="D")
    lons = np.empty((len(days), S))
    lats = np.empty((len(days), S))
    dalt = np.empty((len(days), S))
    for j, day in enumerate(days):
        lons[j], lats[j], dalt[j] = _sample_geometry(sats[which[j]], day + step)

    times = (days.to_numpy()[:, None] + step.to_numpy()[None, :]).astype("datetime64[s]")
    alts = center[:, None, None] + OFFSETS_KM[None, None, :] + dalt[:, :, None]
    shape = (len(days), S, L)
    out = msis.calculate(
        np.broadcast_to(times[:, :, None], shape).ravel(),
        np.broadcast_to(lons[:, :, None], shape).ravel(),
        np.broadcast_to(lats[:, :, None], shape).ravel(),
        np.clip(alts, 80.0, None).ravel(),
    )
    rho = out[:, 0].reshape(shape)
    log_rho = np.log(np.nanmean(rho, axis=1))

    table = pd.DataFrame(log_rho, columns=LEVEL_COLUMNS)
    table.insert(0, "alt_center_km", center)
    table.insert(0, "date", days)
    table.insert(0, "norad_id", norad_id)
    return table


def density_table(
    history: pd.DataFrame,
    samples_per_day: int = 24,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """모든 위성의 밀도 표. cache_dir를 주면 위성별 parquet으로 캐시한다."""
    tables = []
    for norad_id, g in history.groupby("norad_id", sort=True):
        path = None
        if cache_dir is not None:
            key = hashlib.sha256(
                pd.util.hash_pandas_object(
                    g[["epoch", "a_km", "inclination_deg", "raan_deg"]], index=False
                )
                .to_numpy()
                .tobytes()
                + f"{samples_per_day}|{OFFSETS_KM.tolist()}|pad45".encode()
            ).hexdigest()[:16]
            path = Path(cache_dir) / f"density_{int(norad_id)}_{key}.parquet"
            if path.exists():
                tables.append(pd.read_parquet(path))
                continue
        table = satellite_density_table(g, samples_per_day=samples_per_day)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 동기화 폴더에서 반쯤 쓴 파일이 다른 컴퓨터로 넘어가지 않도록 임시 파일에 쓴 뒤 교체
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            table.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


class DensityLookup:
    """(위성, 날짜, 평균고도) → 궤도 평균 밀도. 고도 방향은 ln ρ 선형 보간·외삽."""

    def __init__(self, table: pd.DataFrame):
        self._by_sat = {}
        for norad_id, g in table.groupby("norad_id"):
            g = g.sort_values("date")
            self._by_sat[int(norad_id)] = (
                g["date"].to_numpy(),
                g["alt_center_km"].to_numpy()[:, None] + OFFSETS_KM[None, :],
                g[LEVEL_COLUMNS].to_numpy(),
            )

    def dates(self, norad_id: int) -> np.ndarray:
        return self._by_sat[norad_id][0]

    def rho(self, norad_id: int, dates, alt_km) -> np.ndarray:
        day_index, levels, log_rho = self._by_sat[norad_id]
        dates = np.asarray(dates, dtype="datetime64[ns]").astype("datetime64[D]")
        idx = np.searchsorted(day_index.astype("datetime64[D]"), dates)
        if np.any(idx >= len(day_index)) or np.any(
            day_index.astype("datetime64[D]")[np.clip(idx, 0, len(day_index) - 1)] != dates
        ):
            raise KeyError(f"{norad_id}: 밀도 표에 없는 날짜가 있습니다")
        alt = np.broadcast_to(np.asarray(alt_km, dtype=float), idx.shape)
        lv, lr = levels[idx], log_rho[idx]
        # 구간 선택 (양 끝 밖은 가장자리 구간 기울기로 외삽)
        k = np.clip((lv < alt[:, None]).sum(axis=1) - 1, 0, lv.shape[1] - 2)
        rows = np.arange(len(idx))
        x0, x1 = lv[rows, k], lv[rows, k + 1]
        y0, y1 = lr[rows, k], lr[rows, k + 1]
        return np.exp(y0 + (y1 - y0) * (alt - x0) / (x1 - x0))
