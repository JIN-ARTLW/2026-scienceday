import numpy as np
import pandas as pd

from orbital_decay.selection import CandidateCorrection, available_candidates, select_model


def _results(rmse_by_model: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model, rmse in rmse_by_model.items():
        for i, v in enumerate(rmse):
            rows.append({"norad_id": i, "model": model, "rmse_m_day": v, "fit_seconds": 0.0})
    df = pd.DataFrame(rows)
    phys = df[df["model"] == "physics"].set_index("norad_id")["rmse_m_day"]
    df["skill"] = 1 - df["rmse_m_day"] / df["norad_id"].map(phys)
    return df


def test_selects_clear_winner_and_reports_significance():
    rng = np.random.default_rng(0)
    phys = rng.uniform(5, 10, 15)
    res = _results(
        {
            "physics": phys,
            "fixed_k": phys * 0.97,
            "ridge": phys * rng.uniform(0.95, 1.0, 15),
            "hist_gbm": phys * rng.uniform(0.5, 0.6, 15),
        }
    )
    table, chosen, reasons = select_model(res)
    assert chosen == "hist_gbm"
    assert table.loc["hist_gbm", "p_better_than_fixed_k"] < 0.05


def test_prefers_simpler_model_when_tied():
    rng = np.random.default_rng(1)
    phys = rng.uniform(5, 10, 15)
    noise = rng.normal(0, 0.01, 15)
    res = _results(
        {
            "physics": phys,
            "fixed_k": phys * 0.99,
            "ridge": phys * (0.7 + noise),
            "hist_gbm": phys * (0.7 - noise),
        }
    )
    _, chosen, _ = select_model(res)
    assert chosen == "ridge"


def test_falls_back_to_physics_without_evidence():
    rng = np.random.default_rng(2)
    phys = rng.uniform(5, 10, 12)
    res = _results(
        {
            "physics": phys,
            "fixed_k": phys * rng.uniform(0.9, 1.1, 12),
            "mlp": phys * rng.uniform(0.8, 1.2, 12),
        }
    )
    _, chosen, _ = select_model(res)
    assert chosen == "physics"


def test_candidate_correction_tunes_on_satellite_groups():
    rng = np.random.default_rng(3)
    n = 400
    f107 = rng.uniform(70, 220, n)
    phys = -rng.uniform(0.01, 0.2, n)
    df = pd.DataFrame(
        {
            "norad_id": rng.integers(0, 6, n),
            "is_calib": False,
            "phys_rate_km_day": phys,
            "f107_obs": f107,
        }
    )
    df["obs_rate_km_day"] = phys * (0.8 + 0.004 * (f107 - 70))
    df["ratio"] = df["obs_rate_km_day"] / df["phys_rate_km_day"]
    df["weight"] = phys**2
    ridge = next(c for c in available_candidates() if c.name == "ridge")
    m = CandidateCorrection(ridge, features=["f107_obs"]).fit(df)
    assert m.params_ in ridge.grid
    assert np.allclose(m.correction(df), df["ratio"], atol=0.02)
