import numpy as np
import pandas as pd

from orbital_decay.constants import R_EQ_KM
from orbital_decay.orbit3d import OrbitTrack, earth_vertex_colors, period_minutes


def test_track_geometry_follows_inclination_and_altitude():
    curve = pd.DataFrame(
        {"time": pd.to_datetime(["2024-01-01", "2024-01-11"]), "alt_km": [500.0, 400.0]}
    )
    tr = OrbitTrack(500.0, 51.6, "2024-01-01", curve)
    t = np.linspace(0, 1, 2000)
    xyz, alt = tr.positions(t, "ECI")
    lat_max = np.degrees(np.arcsin(np.abs(xyz[:, 2]) / np.linalg.norm(xyz, axis=1))).max()
    assert abs(lat_max - 51.6) < 0.5
    assert np.allclose(np.linalg.norm(xyz, axis=1), 1 + alt / R_EQ_KM)
    assert tr.altitude(5.0) == 450.0
    # 좌표계 전환은 회전일 뿐 → 거리·z 보존
    ecef, _ = tr.positions(t, "ECEF")
    assert np.allclose(np.linalg.norm(ecef, axis=1), np.linalg.norm(xyz, axis=1))
    assert np.allclose(ecef[:, 2], xyz[:, 2])
    # 고도 과장
    big, _ = tr.positions(t[:1], "ECI", exaggeration=10)
    assert np.isclose(np.linalg.norm(big), 1 + 10 * 500 / R_EQ_KM)


def test_ground_track_on_surface_and_period():
    tr = OrbitTrack(400.0, 97.5, "2024-01-01")
    g = tr.ground_track(0.5, 2, "ECI")
    assert np.allclose(np.linalg.norm(g, axis=1), 1.004)
    assert 92 < period_minutes(400) < 93


def test_earth_colors_without_texture(tmp_path):
    verts = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    cols = earth_vertex_colors(verts, texture=tmp_path / "none.jpg")
    assert cols.shape == (2, 4) and np.all((cols >= 0) & (cols <= 1))
