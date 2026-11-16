import numpy as np
import pandas as pd

from orbital_decay.ashruf import MU, R_MEAN_KM, apparent_altitude, peak_decay_rates


def test_peak_rate_recovers_known_slope_in_m_per_hour():
    t = pd.date_range("1999-01-01", "2003-01-01", freq="13h")
    hours = (t - t[0]).total_seconds() / 3600
    h = 700.0 - 0.5e-3 * hours  # -0.5 m/h
    a_m = (R_MEAN_KM + h) * 1e3
    n_rev_day = np.sqrt(MU / a_m**3) * 86400 / (2 * np.pi)
    el = pd.DataFrame({"norad_id": 1, "epoch": t, "mean_motion": n_rev_day})
    alt = apparent_altitude(el)
    common = pd.DataFrame({"start": [pd.Timestamp("2000-01-01")],
                           "end": [pd.Timestamp("2002-01-01")]}, index=["SC23"])
    rate = peak_decay_rates(alt, common).loc[1, "SC23"]
    assert abs(rate - (-0.5)) < 1e-3
