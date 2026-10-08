"""Ashruf et al. (2026) Table 2 '태양주기 극대기 감쇠율' 재현.

논문 방법 (Section 2, Eq. 1, 7–10):
  1. 평균운동 → 케플러 반장축 a (Eq. 1), 겉보기 고도 h = a − R_E(평균반지름)
  2. 반장축 z-score > 3 인 점을 이상치로 제외
  3. 고도 변화율 Δh (Eq. 7) → 27일 중심 이동평균 (Eq. 8). 고도는 하루 중앙값으로 정리 후 차분
  4. 주기별 임계값 T = μ − kσ (k = 1: SC22·23, 3: SC24) 아래를 급감쇠로 판정 (Eq. 9–10)
  5. 17기의 급감쇠 시작 중 가장 이른 날 ~ 끝 중 가장 늦은 날을 주기별 공통 구간으로 삼고
  6. 그 구간에서 고도의 기울기(m/h)를 구한다.

논문과 다른 점 (재현 한계):
  - 논문은 공통 구간을 그림과 대조해 몇 달씩 수동 조정했다 → 여기서는 자동 구간만 쓴다.
  - 급감쇠 구간은 임계값 아래로 이어지는 가장 긴 구간(15일 이내 끊김은 합침)으로 정한다.
  - 추세가 있는 반장축에 전체 z-score를 쓰면 이상치가 걸리지 않으므로,
    이동중앙값에서 뺀 잔차의 z-score로 판정한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

R_MEAN_KM = 6371.0
MU = 3.986004418e14

CYCLES = {
    "SC22": ("1986-09-01", "1996-08-01", 1.0),
    "SC23": ("1996-08-01", "2008-12-01", 1.0),
    "SC24": ("2008-12-01", "2019-12-01", 3.0),
}


def apparent_altitude(elements: pd.DataFrame) -> pd.DataFrame:
    """Eq. 1: 케플러 반장축과 겉보기 고도, 잔차 z-score 이상치 제거."""
    df = elements[["norad_id", "epoch", "mean_motion"]].dropna().copy()
    df["epoch"] = pd.to_datetime(df["epoch"])
    df = df.sort_values(["norad_id", "epoch"]).drop_duplicates(["norad_id", "epoch"])
    n = df["mean_motion"] * 2 * np.pi / 86400
    df["a_km"] = (MU / n**2) ** (1 / 3) / 1e3
    df["h_km"] = df["a_km"] - R_MEAN_KM
    resid = df["a_km"] - df.groupby("norad_id")["a_km"].transform(
        lambda s: s.rolling(31, center=True, min_periods=5).median()
    )
    z = resid / resid.groupby(df["norad_id"]).transform("std")
    return df[z.abs() <= 3].reset_index(drop=True)


def smoothed_rate(alt: pd.DataFrame, window_days: int = 27) -> pd.DataFrame:
    """Eq. 7–8: 일별 고도 변화율 [m/h]과 27일 중심 이동평균.

    TLE 간격이 몇 시간~며칠로 불규칙해 인접 TLE 차분은 잡음이 매우 크다.
    Eq. 8이 27개 점 = 27일을 전제하므로 고도를 하루 중앙값으로 정리한 뒤 차분한다.
    """
    out = []
    for nid, g in alt.groupby("norad_id"):
        daily = g.set_index("epoch")["h_km"].resample("1D").median().dropna()
        dt_h = daily.index.to_series().diff().dt.total_seconds() / 3600
        rate = (daily.diff() * 1e3 / dt_h).iloc[1:]
        smooth = rate.rolling(f"{window_days}D", center=True, min_periods=10).mean()
        out.append(
            pd.DataFrame(
                {"norad_id": nid, "epoch": rate.index, "rate": rate.values, "smooth": smooth.values}
            )
        )
    return pd.concat(out, ignore_index=True)


def rapid_windows(rates: pd.DataFrame, method: str = "span") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eq. 9–10: 위성·주기별 급감쇠 시작/끝, 그리고 주기별 공통 구간.

    method="span": 임계값을 처음 넘는 날 ~ 마지막으로 넘는 날 (논문 서술 그대로)
    method="longest": 임계값 아래로 이어지는 가장 긴 구간 (잠깐 넘는 잡음에 덜 민감)
    """
    rows = []
    for cyc, (a, b, k) in CYCLES.items():
        r = rates[(rates["epoch"] >= a) & (rates["epoch"] < b)]
        for nid, g in r.groupby("norad_id"):
            g = g.dropna(subset=["smooth"]).reset_index(drop=True)
            thr = g["smooth"].mean() - k * g["smooth"].std()
            below = (g["smooth"] < thr).to_numpy()
            if not below.any():
                continue
            # 임계값 아래로 '이어지는' 구간들 중 가장 긴 것 (잡음으로 잠깐 넘는 점 제외).
            # 15일 이내 끊김은 같은 구간으로 본다.
            run_id = np.cumsum(np.r_[True, np.diff(below.astype(int)) != 0])
            runs = [
                (g["epoch"][run_id == r].min(), g["epoch"][run_id == r].max())
                for r in np.unique(run_id[below])
            ]
            merged = [list(runs[0])]
            for a_, b_ in runs[1:]:
                if (a_ - merged[-1][1]).days <= 15:
                    merged[-1][1] = b_
                else:
                    merged.append([a_, b_])
            if method == "span":
                start, end = merged[0][0], merged[-1][1]
            else:
                start, end = max(merged, key=lambda r: r[1] - r[0])
            rows.append(
                {"cycle": cyc, "norad_id": nid, "start": start, "end": end, "threshold_m_h": thr}
            )
    per_obj = pd.DataFrame(rows)
    common = per_obj.groupby("cycle").agg(start=("start", "min"), end=("end", "max"))
    return per_obj, common


def peak_decay_rates(alt: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    """공통 구간 안의 고도 기울기 [m/h] (위성 × 주기)."""
    rows = []
    for cyc, w in common.iterrows():
        m = (alt["epoch"] >= w["start"]) & (alt["epoch"] <= w["end"])
        for nid, g in alt[m].groupby("norad_id"):
            if len(g) < 10:
                continue
            t_h = (g["epoch"] - g["epoch"].iloc[0]).dt.total_seconds() / 3600
            slope = np.polyfit(t_h, g["h_km"] * 1e3, 1)[0]
            rows.append({"norad_id": nid, "cycle": cyc, "rate_m_h": slope, "n_tle": len(g)})
    return pd.DataFrame(rows).pivot(index="norad_id", columns="cycle", values="rate_m_h")


# 논문 Table 2 (m/h)
PAPER_TABLE2 = pd.DataFrame(
    {
        "SC22": [
            -1.42,
            -0.52,
            -0.39,
            -0.73,
            -0.90,
            -0.24,
            -0.42,
            -1.04,
            -1.15,
            -0.53,
            -0.66,
            -0.47,
            -0.52,
            -0.25,
            -0.17,
            -0.36,
            -0.32,
        ],
        "SC23": [
            -1.40,
            -0.46,
            -0.30,
            -0.61,
            -0.93,
            -0.18,
            -0.33,
            -1.20,
            -1.10,
            -0.45,
            -0.63,
            -0.40,
            -0.43,
            -0.18,
            -0.12,
            -0.27,
            -0.25,
        ],
        "SC24": [
            -0.81,
            -0.17,
            -0.11,
            -0.23,
            -0.44,
            -0.06,
            -0.11,
            -0.71,
            -0.63,
            -0.16,
            -0.27,
            -0.15,
            -0.15,
            -0.07,
            -0.04,
            -0.09,
            -0.09,
        ],
    },
    index=pd.Index(
        [22, 29, 45, 46, 115, 162, 227, 228, 262, 309, 397, 716, 720, 733, 734, 876, 877],
        name="norad_id",
    ),
)
