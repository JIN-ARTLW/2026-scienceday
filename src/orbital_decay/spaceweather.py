"""일별 태양·지자기 지수 표.

MSIS와 같은 자료원(CelesTrak SW-All.csv, pymsis가 자동으로 받아 캐시)을 써서
물리모델 입력과 ML 특징이 서로 다른 F10.7 정의를 섞지 않게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymsis import msis


def daily_space_weather(start, end, pad_days: int = 100) -> pd.DataFrame:
    """[start, end] 날짜별 지수 표. 후행 평균 계산을 위해 앞쪽을 pad_days만큼 더 읽는다.

    열:
      f107_obs        그날 관측 F10.7 (보정값 아님, MSIS 입력과 같은 관측값)
      f107a_centered  81일 중심 평균 (MSIS 입력용; 미래 40일을 포함하므로 ML 특징에서는 제외)
      f107_trail27/81 과거 27/81일 후행 평균 (미래 정보 없음)
      ap_daily        일평균 Ap
      ap_trail3       과거 3일 Ap 평균
      ap_max3         과거 3일 3시간 ap 최댓값
    """
    start = pd.Timestamp(start).floor("D")
    end = pd.Timestamp(end).floor("D")
    days = pd.date_range(start - pd.Timedelta(days=pad_days), end, freq="D")

    # MSIS 관례: 날짜 D의 f107 입력은 D-1의 관측값이다. 따라서 D+1로 조회하면 D의 관측값.
    next_day = (days + pd.Timedelta(days=1)).to_numpy().astype("datetime64[s]")
    f107_obs, _, _ = msis.get_f107_ap(next_day)
    noon = (days + pd.Timedelta(hours=12)).to_numpy().astype("datetime64[s]")
    _, f107a, ap = msis.get_f107_ap(noon)

    sw = pd.DataFrame(
        {
            "date": days,
            "f107_obs": f107_obs,
            "f107a_centered": f107a,
            "ap_daily": ap[:, 0],
        }
    )
    # 3시간 ap 8개로 그날 최댓값을 만들기 위해 3시간 간격 조회
    three_hourly = days.repeat(8) + pd.to_timedelta(np.tile(np.arange(8) * 3, len(days)), "h")
    _, _, ap3 = msis.get_f107_ap(three_hourly.to_numpy().astype("datetime64[s]"))
    sw["ap_max_day"] = ap3[:, 1].reshape(-1, 8).max(axis=1)

    sw["f107_trail27"] = sw["f107_obs"].rolling(27, min_periods=20).mean()
    sw["f107_trail81"] = sw["f107_obs"].rolling(81, min_periods=60).mean()
    sw["ap_trail3"] = sw["ap_daily"].rolling(3, min_periods=1).mean()
    sw["ap_max3"] = sw["ap_max_day"].rolling(3, min_periods=1).max()
    return sw[sw["date"] >= start].reset_index(drop=True)
