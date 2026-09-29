"""모델 비교: 위성 단위 분할, 감쇠율 오차, 태양활동 조건별 오차, 궤도 전파 오차."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from orbital_decay.constants import R_EQ_KM
from orbital_decay.density import DensityLookup
from orbital_decay.models import PhysicsModel
from orbital_decay.physics import drag_factor, propagate


def split_satellites(norad_ids, test_fraction: float = 1 / 3, seed: int = 0):
    """위성 ID를 학습/검증으로 나눈다 (시간이 아니라 위성 단위)."""
    ids = np.array(sorted(set(int(i) for i in norad_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, round(len(ids) * test_fraction))
    return sorted(ids[n_test:].tolist()), sorted(ids[:n_test].tolist())


def rate_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """감쇠율 오차. 단위는 m/day (감쇠는 음수)."""
    obs, pred = np.asarray(obs) * 1e3, np.asarray(pred) * 1e3
    err = pred - obs
    both_neg = (obs < 0) & (pred < 0)
    log_err = np.log(pred[both_neg] / obs[both_neg])
    ss_res = np.sum(err**2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return {
        "n": len(obs),
        "rmse_m_day": float(np.sqrt(np.mean(err**2))),
        "mae_m_day": float(np.mean(np.abs(err))),
        "bias_m_day": float(np.mean(err)),
        "median_abs_pct": float(np.median(np.abs(np.expm1(log_err))) * 100),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def _eval_rows(df: pd.DataFrame, ids) -> pd.DataFrame:
    return df[df["norad_id"].isin(ids) & ~df["is_calib"]]


def compare_models(df: pd.DataFrame, train_ids, test_ids, models: list[PhysicsModel]):
    """학습 위성으로 fit, 검증 위성의 평가 구간에서 감쇠율 오차를 비교한다.

    반환: (요약 표, 창별 예측 표)
    """
    train = df[df["norad_id"].isin(train_ids)]
    test = _eval_rows(df, test_ids).copy()
    fitted, summary = [], []
    for model in models:
        model = copy.deepcopy(model).fit(train)
        fitted.append(model)
        test[f"pred_{model.name}"] = model.predict_rate(test)
        summary.append(
            {
                "model": model.name,
                **rate_metrics(test["obs_rate_km_day"], test[f"pred_{model.name}"]),
            }
        )
    return pd.DataFrame(summary).set_index("model"), test, fitted


def leave_one_satellite_out(df: pd.DataFrame, models: list[PhysicsModel]) -> pd.DataFrame:
    """위성 하나씩 빼서 검증 (위성 수가 적을 때 2/3·1/3 분할보다 안정적)."""
    ids = sorted(df["norad_id"].unique())
    preds = []
    for held_out in ids:
        _, test, _ = compare_models(df, [i for i in ids if i != held_out], [held_out], models)
        preds.append(test)
    return pd.concat(preds, ignore_index=True)


def summarize(
    pred: pd.DataFrame, models: list[PhysicsModel], by: str | None = None
) -> pd.DataFrame:
    """창별 예측 표 → 모델별 오차 (by 열로 그룹별)."""
    groups = [(None, pred)] if by is None else pred.groupby(by, observed=True)
    rows = []
    for key, g in groups:
        for m in models:
            rows.append(
                {
                    **({by: key} if by else {}),
                    "model": m.name,
                    **rate_metrics(g["obs_rate_km_day"], g[f"pred_{m.name}"]),
                }
            )
    out = pd.DataFrame(rows)
    return out.set_index([by, "model"] if by else "model")


def activity_bins(pred: pd.DataFrame, col: str = "f107_trail81") -> pd.Series:
    """태양활동 조건 구분: 81일 후행 평균 F10.7 기준 낮음(<100)/보통/높음(>=150)."""
    return pd.cut(pred[col], [0, 100, 150, np.inf], labels=["low(<100)", "mid", "high(>=150)"])


def correction_function(
    model, sat_rows: pd.DataFrame, space_weather: pd.DataFrame, lookup: DensityLookup
):
    """궤도 전파에 넣을 감쇠율 배율 함수 (날짜, 고도) → 배율.

    위성의 정적 특징(경사각, 이심률, BC)은 보정 구간 마지막 값을 쓰고,
    고도와 물리 예측 감쇠율은 전파 중 예측 궤도에서 다시 계산한다 (관측 TLE는 쓰지 않음).
    """
    if type(model) is PhysicsModel:
        return None
    last = sat_rows[sat_rows["is_calib"]].iloc[-1]
    sw = space_weather.set_index(
        space_weather["date"].dt.floor("D").to_numpy().astype("datetime64[D]")
    )
    norad_id = int(last["norad_id"])

    def fn(days: np.ndarray, alt_km: np.ndarray) -> np.ndarray:
        row = sw.loc[days].reset_index(drop=True)
        rho = lookup.rho(norad_id, days, alt_km)
        row["alt_mid_km"] = alt_km
        row["inc_deg"] = last["inc_deg"]
        row["ecc"] = last["ecc"]
        row["log_bc"] = last["log_bc"]
        row["log_phys_rate"] = np.log(-last["bc_est"] * drag_factor(rho, alt_km + R_EQ_KM))
        if "area_to_mass" in sat_rows:
            row["area_to_mass"] = last["area_to_mass"]
        return model.correction(row)

    return fn


def propagation_errors(
    df: pd.DataFrame,
    test_ids,
    fitted_models: list[PhysicsModel],
    space_weather: pd.DataFrame,
    lookup: DensityLookup,
    horizons=(30, 90, 180, 365),
):
    """보정 구간 끝에서 출발해 각 모델로 고도를 전파하고, 이후 TLE 고도와 비교한다.

    반환: (horizon별 고도 오차 표 [km], 전파 곡선 표)
    """
    rows, curves = [], []
    for norad_id in test_ids:
        sat = df[df["norad_id"] == norad_id].sort_values("t_mid")
        ev = sat[~sat["is_calib"]]
        if ev.empty:
            continue
        t0, a0 = ev["t_start"].iloc[0], ev["a_start_km"].iloc[0]
        obs_t = (ev["t_mid"] - t0).dt.total_seconds().to_numpy() / 86400
        n_days = int(min(max(horizons), obs_t.max()))
        for model in fitted_models:
            curve = propagate(
                norad_id,
                a0,
                t0,
                n_days,
                float(sat["bc_est"].iloc[0]),
                lookup,
                correction=correction_function(model, sat, space_weather, lookup),
            )
            curve["norad_id"], curve["model"] = norad_id, model.name
            curves.append(curve)
            ct = (curve["time"] - t0).dt.total_seconds().to_numpy() / 86400
            for h in horizons:
                if h > obs_t.max() or h > ct.max():
                    continue
                obs_a = np.interp(h, obs_t, ev["a_mid_km"])
                rows.append(
                    {
                        "norad_id": norad_id,
                        "model": model.name,
                        "horizon_days": h,
                        "alt_error_km": float(np.interp(h, ct, curve["a_km"]) - obs_a),
                    }
                )
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
