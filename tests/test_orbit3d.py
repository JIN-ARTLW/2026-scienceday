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


# sgp4 문서에 공개된 ISS 예시 TLE
ISS = (
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482",
)


def test_viewer_frame_matches_skyfield_subpoint():
    """뷰어의 TEME→지구고정 변환 + 측지위도 보정이 skyfield 직하점과 일치하는지."""
    from sgp4.api import Satrec
    from skyfield.api import EarthSatellite, load, wgs84

    from orbital_decay.constants import WGS84_F
    from orbital_decay.orbit3d import _UNIX_JD, gmst_rad, rotate_z

    sat = Satrec.twoline2rv(*ISS)
    ts = load.timescale()
    es = EarthSatellite(*ISS, "ISS", ts)
    times = pd.date_range(es.epoch.utc_datetime(), periods=200, freq="11min").tz_localize(None)
    jd_full = times.to_numpy().astype("datetime64[ns]").astype(np.int64) / 86400e9 + _UNIX_JD
    jd = np.floor(jd_full)
    _, r, _ = sat.sgp4_array(jd, jd_full - jd)
    ecef = rotate_z(r, -gmst_rad(times))
    lon = np.degrees(np.arctan2(ecef[:, 1], ecef[:, 0]))
    psi = np.arcsin(ecef[:, 2] / np.linalg.norm(ecef, axis=1))
    lat = np.degrees(np.arctan(np.tan(psi) / (1 - WGS84_F) ** 2))
    t = ts.utc(
        *(getattr(times, a).to_numpy() for a in ("year", "month", "day", "hour", "minute")),
        (times.second + times.microsecond / 1e6).to_numpy(),
    )
    sp = wgs84.subpoint_of(es.at(t))
    assert np.abs((lon - sp.longitude.degrees + 180) % 360 - 180).max() < 0.01
    assert np.abs(lat - sp.latitude.degrees).max() < 0.03


def test_texture_registration_landmarks():
    """지구 텍스처의 대륙이 실제 위경도에 있는지: 작은 섬·좁은 바다 포함 기준점 판별."""
    import pytest

    from orbital_decay.orbit3d import TEXTURE_PATH

    if not TEXTURE_PATH.exists():
        pytest.skip("assets/earth.jpg 없음")
    land = [
        (23, 10),
        (-25, 134),
        (7.9, 80.7),
        (65, -18.5),
        (21.8, -79.5),
        (36.5, 127.8),
        (-42, 146.6),
        (19.6, -155.5),
        (-44, 170.5),
        (42.5, 12.8),
        (-80, 0),
        (72, -40),
    ]
    water = [
        (0, -150),
        (35, 18),
        (43, 34),
        (40, 134),
        (36, 123.8),
        (20, 38.5),
        (27, 51.5),
        (34.5, 129),
        (60, -85),
        (-18, 41),
        (42, 50.5),
        (15, 88),
    ]
    pts = np.array(land + water, dtype=float)
    lat, lon = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    # 측지위도 → 구 위 방향(지심위도)으로 바꿔 뷰어와 같은 경로로 색을 읽는다
    from orbital_decay.constants import WGS84_F

    psi = np.arctan(np.tan(lat) * (1 - WGS84_F) ** 2)
    verts = np.column_stack([np.cos(psi) * np.cos(lon), np.cos(psi) * np.sin(lon), np.sin(psi)])
    rgb = earth_vertex_colors(verts)[:, :3] * 255
    is_water = (rgb[:, 2] > rgb[:, 0] + 15) & (rgb[:, 2] >= rgb[:, 1])
    assert not is_water[: len(land)].any()
    assert is_water[len(land) :].all()
