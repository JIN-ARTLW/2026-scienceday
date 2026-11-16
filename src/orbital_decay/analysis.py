"""태양활동과 감쇠의 상관·지연 분석."""

from __future__ import annotations

import numpy as np
import pandas as pd

from orbital_decay.constants import MU_EARTH, SECONDS_PER_DAY


def tle_density(df: pd.DataFrame) -> pd.Series:
    """TLE 감쇠율에서 역산한 유효 궤도평균 밀도 ρ_eff [kg/m^3] = -(da/dt) / (BC sqrt(μa)).

    고도와 BC 차이를 나눠 없앤 값이라 위성끼리, 태양활동 지수와 비교하기 좋다.
    """
    rate_m_s = df["obs_rate_km_day"] * 1e3 / SECONDS_PER_DAY
    rho = -rate_m_s / (df["bc_est"] * np.sqrt(MU_EARTH * df["a_mid_km"] * 1e3))
    return rho.where(rho > 0)


def lag_correlation(
    df: pd.DataFrame,
    space_weather: pd.DataFrame,
    index: str = "f107_obs",
    max_lag_days: int = 30,
    value: pd.Series | None = None,
) -> pd.DataFrame:
    """ln ρ_eff(t) 와 index(t - lag)의 피어슨 상관을 lag별로 계산.

    위성마다 장기 추세(고도 변화, 태양주기)를 없애기 위해 ln ρ_eff와 지수 모두에서
    약 1년 이동중앙값을 뺀 뒤 계산한다 → 27일 자전 주기·폭풍 같은 단기 변동의 지연을 본다.
    최대 상관이 나타나는 lag가 대기 반응 지연의 추정치이다 (창 길이만큼 흐려진다는 점에 주의).
    """
    d = df[["norad_id", "t_mid"]].copy()
    d["y"] = np.log(tle_density(df) if value is None else value)
    d = d.dropna().sort_values(["norad_id", "t_mid"])
    d["y"] = d["y"] - d.groupby("norad_id")["y"].transform(
        lambda s: s.rolling(52, center=True, min_periods=10).median()
    )
    sw = space_weather.set_index("date")[index]
    sw = sw - sw.rolling(365, center=True, min_periods=180).median()  # 같은 방식으로 장기 추세 제거
    base = d["t_mid"].dt.floor("D")
    rows = []
    for lag in range(0, max_lag_days + 1):
        x = sw.reindex(base - pd.Timedelta(days=lag)).to_numpy()
        ok = np.isfinite(x) & np.isfinite(d["y"].to_numpy())
        rows.append(
            {
                "lag_days": lag,
                "r": np.corrcoef(x[ok], d["y"].to_numpy()[ok])[0, 1],
                "n": int(ok.sum()),
            }
        )
    return pd.DataFrame(rows)
