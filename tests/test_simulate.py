import warnings

import joblib
import numpy as np
import pandas as pd

from orbital_decay.constants import R_EQ_KM
from orbital_decay.models import FixedCorrectionModel, MLCorrectionModel, freeze
from orbital_decay.physics import REENTRY_KM, propagate
from orbital_decay.simulate import GridDensity, Scenario, simulate, space_weather_end

warnings.filterwarnings("ignore", message="There is data that was either interpolated")


def _exponential_grid(days, rho0=1e-11, h0=400.0, H=50.0):
    levels = np.arange(80.0, 600.0, 20.0)
    log_rho = np.log(rho0) - (levels - h0) / H
    return GridDensity(days, levels, np.tile(log_rho, (len(days), 1)))


def test_propagate_reaches_reentry_without_stalling():
    days = pd.date_range("2020-01-01", periods=4000)
    grid = _exponential_grid(days)
    out = propagate(0, R_EQ_KM + 300, days[0], 3999, 0.02, grid, steps_per_day=1)
    assert out["alt_km"].iloc[-1] < REENTRY_KM
    assert out["time"].iloc[-1] < days[-1]  # 기간 끝까지 가지 않고 재진입에서 멈춤
    # 간격 설정과 무관하게 같은 재진입 시각
    out4 = propagate(0, R_EQ_KM + 300, days[0], 3999, 0.02, grid, steps_per_day=4)
    t1, t4 = out["time"].iloc[-1], out4["time"].iloc[-1]
    assert abs((t1 - t4).total_seconds()) < 86400


def test_simulate_short_scenario_and_models():
    train = pd.DataFrame(
        {
            "norad_id": [1] * 40,
            "is_calib": False,
            "phys_rate_km_day": -0.05,
            "obs_rate_km_day": -0.06,
            "weight": 0.0025,
        }
    )
    train["ratio"] = train["obs_rate_km_day"] / train["phys_rate_km_day"]
    k = FixedCorrectionModel().fit(train)
    curves, info = simulate(
        Scenario(alt0_km=300, bc=0.02, start="2014-01-01", days=120),
        {"physics": None, "fixed_k": k},
    )
    last = curves.groupby("model")["alt_km"].last()
    assert last["fixed_k"] < last["physics"] < 300  # 배율 1.2 → 더 빨리 감쇠
    assert info["uses_analog_after"] is None


def test_future_scenario_uses_analog_cycle():
    start = space_weather_end() - pd.Timedelta(days=10)
    _, info = simulate(Scenario(alt0_km=500, bc=0.005, start=f"{start:%Y-%m-%d}", days=60))
    assert info["uses_analog_after"] == space_weather_end()


def test_frozen_model_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {"norad_id": rng.integers(0, 4, n), "is_calib": False, "f107_obs": rng.uniform(70, 200, n)}
    )
    df["phys_rate_km_day"] = -0.05
    df["obs_rate_km_day"] = -0.05 * (0.8 + 0.003 * df["f107_obs"])
    df["ratio"] = df["obs_rate_km_day"] / df["phys_rate_km_day"]
    df["weight"] = 0.0025
    ml = MLCorrectionModel(features=["f107_obs"], backend="sklearn").fit(df)
    joblib.dump({"ml": freeze(ml)}, tmp_path / "m.joblib")
    loaded = joblib.load(tmp_path / "m.joblib")["ml"]
    assert np.allclose(loaded.correction(df), ml.correction(df))
