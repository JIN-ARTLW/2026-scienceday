import warnings

import numpy as np
import pandas as pd
import pytest

from orbital_decay.constants import MU_EARTH, R_EQ_KM
from orbital_decay.density import LEVEL_COLUMNS, OFFSETS_KM, DensityLookup
from orbital_decay.evaluation import rate_metrics, split_satellites
from orbital_decay.models import FixedCorrectionModel, MLCorrectionModel, PhysicsModel
from orbital_decay.orbit import decay_windows, prepare_history
from orbital_decay.physics import drag_factor, estimate_bc, propagate

warnings.filterwarnings("ignore", message="There is data that was either interpolated")


def _elements(a_km, epochs, norad_id=1):
    n = np.sqrt(MU_EARTH / (np.asarray(a_km) * 1e3) ** 3) * 86400 / (2 * np.pi)
    return pd.DataFrame(
        {
            "norad_id": norad_id,
            "epoch": epochs,
            "mean_motion": n,
            "eccentricity": 0.001,
            "inclination_deg": 51.6,
            "raan_deg": 10.0,
            "arg_pericenter_deg": 0.0,
            "mean_anomaly_deg": 0.0,
            "bstar": 1e-4,
        }
    )


def _constant_lookup(norad_id, days, log_rho_at_center, center_alt, scale_height_km=60.0):
    """고도에 대해 지수형(ln ρ 선형) 밀도를 가진 표."""
    table = pd.DataFrame({"norad_id": norad_id, "date": days, "alt_center_km": center_alt})
    for col, off in zip(LEVEL_COLUMNS, OFFSETS_KM, strict=True):
        table[col] = log_rho_at_center - off / scale_height_km
    return DensityLookup(table)


def test_drag_factor_magnitude():
    # 400 km, ρ=1e-12, BC=0.01 → 약 45 m/day 감쇠
    rate_m_day = 0.01 * drag_factor(1e-12, R_EQ_KM + 400) * 1e3
    assert -50 < rate_m_day < -40


def test_estimate_bc_exact():
    g = -np.linspace(1, 3, 20)
    assert estimate_bc(0.012 * g, g) == pytest.approx(0.012)


def test_prepare_history_and_windows_recover_slope_and_drop_outlier():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 60, 120))
    a = R_EQ_KM + 450 - 0.03 * t + rng.normal(0, 0.02, t.size)
    a[40] += 4.0  # 이상치
    epochs = pd.Timestamp("2020-01-01") + pd.to_timedelta(t, unit="D")
    hist = prepare_history(_elements(a, epochs))
    assert hist.attrs["cleaning"][1]["outliers"] >= 1
    assert hist["a_km"].max() < R_EQ_KM + 455
    w = decay_windows(hist, window_days=10)
    assert w["obs_rate_km_day"].median() == pytest.approx(-0.03, abs=0.01)


def test_density_lookup_log_linear_interpolation():
    days = pd.date_range("2020-01-01", periods=3)
    lk = _constant_lookup(7, days, np.log(1e-12), 400.0, scale_height_km=50.0)
    rho = lk.rho(7, days[:1], [450.0])
    assert rho[0] == pytest.approx(1e-12 * np.exp(-1), rel=1e-9)
    # 표 범위 밖 외삽도 같은 척도고도
    assert lk.rho(7, days[:1], [600.0])[0] == pytest.approx(1e-12 * np.exp(-4), rel=1e-9)


def test_propagate_matches_analytic_constant_density():
    days = pd.date_range("2020-01-01", periods=40)
    # 척도고도를 매우 크게 → 고도 무관 일정 밀도
    lk = _constant_lookup(3, days, np.log(2e-12), 400.0, scale_height_km=1e9)
    bc, a0 = 0.02, R_EQ_KM + 400
    out = propagate(3, a0, days[0], 30, bc, lk)
    # d(√a)/dt = -BC ρ √μ / 2  (a는 m)
    k = bc * 2e-12 * np.sqrt(MU_EARTH) / 2 * 86400
    expected = (np.sqrt(a0 * 1e3) - k * 30) ** 2 / 1e3
    assert out["a_km"].iloc[-1] == pytest.approx(expected, abs=1e-3)


def _model_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    f107 = rng.uniform(70, 220, n)
    phys = -rng.uniform(0.01, 0.2, n)
    true_ratio = 0.8 + 0.004 * (f107 - 70)
    df = pd.DataFrame(
        {
            "norad_id": rng.integers(0, 6, n),
            "is_calib": False,
            "phys_rate_km_day": phys,
            "obs_rate_km_day": phys * true_ratio + rng.normal(0, 0.002, n),
            "alt_mid_km": rng.uniform(350, 600, n),
            "inc_deg": 51.6,
            "ecc": 0.001,
            "log_bc": np.log(0.01),
            "log_phys_rate": np.log(-phys),
            "f107_obs": f107,
            "f107_trail27": f107,
            "f107_trail81": f107,
            "ap_daily": 5.0,
            "ap_trail3": 5.0,
            "ap_max3": 7.0,
        }
    )
    df["ratio"] = df["obs_rate_km_day"] / df["phys_rate_km_day"]
    df["weight"] = df["phys_rate_km_day"] ** 2
    return df, true_ratio


def test_fixed_k_recovers_mean_scale():
    df, true_ratio = _model_frame()
    k = FixedCorrectionModel().fit(df).k_
    assert 0.8 < k < 1.5
    assert PhysicsModel().fit(df).correction(df) == pytest.approx(np.ones(len(df)))


def test_ml_learns_activity_dependent_correction():
    train, _ = _model_frame(seed=1)
    test, true_ratio = _model_frame(seed=2)
    ml = MLCorrectionModel(backend="sklearn").fit(train)
    fixed = FixedCorrectionModel().fit(train)
    err_ml = rate_metrics(test["obs_rate_km_day"], ml.predict_rate(test))["rmse_m_day"]
    err_k = rate_metrics(test["obs_rate_km_day"], fixed.predict_rate(test))["rmse_m_day"]
    assert err_ml < 0.6 * err_k
    assert np.corrcoef(ml.correction(test), true_ratio)[0, 1] > 0.9


def test_split_satellites_disjoint_and_complete():
    train, test = split_satellites(range(12), 1 / 3, seed=3)
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(12))
    assert len(test) == 4


def test_end_to_end_synthetic_recovers_bc():
    from orbital_decay.dataset import build_dataset
    from orbital_decay.density import density_table
    from orbital_decay.spaceweather import daily_space_weather
    from orbital_decay.synthetic import simulate_satellite

    rng = np.random.default_rng(5)
    sw = daily_space_weather("2015-01-01", "2015-06-01")
    el, truth = simulate_satellite(
        1,
        "2015-01-01",
        120,
        420.0,
        51.6,
        0.01,
        sw,
        rng,
        bias=lambda *a: 1.0,
        samples_per_day=8,
        outlier_rate=0.0,
    )
    hist = prepare_history(el)
    lk = DensityLookup(density_table(hist, samples_per_day=8))
    df = build_dataset(decay_windows(hist, 14), lk, sw, calib_days=60, min_calib_windows=3)
    # 기하·표본 차이만 있으므로 BC는 정답과 10% 안에서 복원돼야 한다
    assert df["bc_est"].iloc[0] == pytest.approx(0.01, rel=0.1)
