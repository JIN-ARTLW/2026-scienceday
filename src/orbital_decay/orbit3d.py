"""GMAT 스타일 3D 궤도 화면 (뷰어의 '3D 궤도' 탭).

- 단위: 지구 적도반지름 = 1. 관성좌표(ECI, 지구가 GMST로 자전) / 지구고정(ECEF) 전환.
- 궤도 형상: 시나리오의 원궤도를 항력 없는 SGP4로 전파 (경사각, J2 교점 세차 포함).
- 고도: 시뮬레이션 결과(모델별 평균고도 곡선)를 따라 줄어든다. 고도 과장 배율로 확대해 볼 수 있다.
- 지구 텍스처: assets/earth.jpg(정거원통도법, 가로:세로 = 2:1)가 있으면 입히고,
  없으면 바다색 구와 위경도 격자만 그린다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from orbital_decay.constants import R_EQ_KM
from orbital_decay.orbit import make_satrec

TEXTURE_PATH = Path(__file__).resolve().parents[2] / "assets" / "earth.jpg"
_UNIX_JD = 2440587.5
SAMPLES_PER_ORBIT = 180


def gmst_rad(times: pd.DatetimeIndex) -> np.ndarray:
    jd = times.to_numpy().astype("datetime64[ns]").astype(np.int64) / 86400e9 + _UNIX_JD
    return np.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)


class OrbitTrack:
    """시나리오 궤도의 3D 좌표 계산 (Qt와 무관, 테스트 가능)."""

    def __init__(self, alt0_km: float, inc_deg: float, start, curve: pd.DataFrame | None = None):
        self.alt0_km = alt0_km
        self.start = pd.Timestamp(start)
        r = R_EQ_KM + alt0_km
        n_rev_day = np.sqrt(3.986004418e14 / (r * 1e3) ** 3) * 86400 / (2 * np.pi)
        self.period_min = 1440.0 / n_rev_day
        self.sat = make_satrec(
            pd.Series(
                {
                    "epoch": self.start,
                    "mean_motion": n_rev_day,
                    "eccentricity": 0.0005,
                    "inclination_deg": inc_deg,
                    "raan_deg": 0.0,
                    "arg_pericenter_deg": 0.0,
                    "mean_anomaly_deg": 0.0,
                    "bstar": 0.0,
                }
            )
        )
        if curve is None or curve.empty:
            self.t_days = np.array([0.0, 1.0])
            self.alt = np.array([alt0_km, alt0_km])
        else:
            self.t_days = (curve["time"] - self.start).dt.total_seconds().to_numpy() / 86400
            self.alt = curve["alt_km"].to_numpy()

    @property
    def duration_days(self) -> float:
        return float(self.t_days[-1])

    def altitude(self, t_days) -> np.ndarray:
        return np.interp(t_days, self.t_days, self.alt)

    def positions(self, t_days: np.ndarray, frame: str = "ECI", exaggeration: float = 1.0):
        """t_days(시작 후 일) 시각들의 위치 [지구반지름 단위]. 반환: (N,3) 좌표, (N,) 고도 km."""
        times = self.start + pd.to_timedelta(np.asarray(t_days), unit="D")
        jd_full = times.to_numpy().astype("datetime64[ns]").astype(np.int64) / 86400e9 + _UNIX_JD
        jd = np.floor(jd_full)
        _, r, _ = self.sat.sgp4_array(jd, jd_full - jd)
        direction = r / np.linalg.norm(r, axis=1, keepdims=True)
        alt = self.altitude(t_days)
        xyz = direction * (1.0 + exaggeration * alt / R_EQ_KM)[:, None]
        if frame == "ECEF":
            xyz = rotate_z(xyz, -gmst_rad(pd.DatetimeIndex(times)))
        return xyz, alt

    def trail(self, t_days: float, n_orbits: float, frame: str, exaggeration: float):
        span = n_orbits * self.period_min / 1440.0
        t = np.linspace(max(0.0, t_days - span), t_days, max(2, int(SAMPLES_PER_ORBIT * n_orbits)))
        return self.positions(t, frame, exaggeration)[0]

    def ground_track(self, t_days: float, n_orbits: float, frame: str, height: float = 1.004):
        """지표면에 투영한 궤적 (ECI 화면에서는 지구 자전을 반영해 현재 지구 위에 그림)."""
        span = n_orbits * self.period_min / 1440.0
        t = np.linspace(max(0.0, t_days - span), t_days, max(2, int(SAMPLES_PER_ORBIT * n_orbits)))
        xyz, _ = self.positions(t, "ECEF", 0.0)
        xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True) * height
        if frame == "ECI":
            now = pd.DatetimeIndex([self.start + pd.Timedelta(days=t_days)])
            xyz = rotate_z(xyz, np.full(len(xyz), gmst_rad(now)[0]))
        return xyz


def period_minutes(alt_km: float) -> float:
    a = (R_EQ_KM + alt_km) * 1e3
    return float(2 * np.pi * np.sqrt(a**3 / 3.986004418e14) / 60)


def rotate_z(xyz: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x, y = xyz[:, 0], xyz[:, 1]
    return np.column_stack([c * x - s * y, s * x + c * y, xyz[:, 2]])


def earth_vertex_colors(verts: np.ndarray, texture: Path = TEXTURE_PATH) -> np.ndarray:
    """구 꼭짓점(지구고정 좌표)의 색. 텍스처가 없으면 바다색에 위도 음영."""
    lat = np.degrees(np.arcsin(np.clip(verts[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(verts[:, 1], verts[:, 0]))
    if texture.exists():
        from PySide6.QtGui import QImage

        img = QImage(str(texture))
        if not img.isNull():
            w, h = img.width(), img.height()
            px = np.clip(((lon + 180) / 360 * w).astype(int), 0, w - 1)
            py = np.clip(((90 - lat) / 180 * h).astype(int), 0, h - 1)
            cols = np.array(
                [img.pixelColor(int(x), int(y)).getRgbF() for x, y in zip(px, py, strict=True)]
            )
            return cols
    shade = 0.75 + 0.25 * np.cos(np.radians(lat))
    base = np.array([0.16, 0.38, 0.66, 1.0])
    cols = np.tile(base, (len(verts), 1))
    cols[:, :3] *= shade[:, None]
    return cols


def graticule(step_deg: float = 30.0, radius: float = 1.002):
    """위경도 격자선 목록 (N,3) 배열들. 적도·본초자오선은 따로 표시."""
    lines = []
    t = np.radians(np.linspace(-180, 180, 361))
    for lat in np.arange(-90 + step_deg, 90, step_deg):
        la = np.radians(lat)
        lines.append(
            (
                lat == 0,
                np.column_stack(
                    [np.cos(la) * np.cos(t), np.cos(la) * np.sin(t), np.full_like(t, np.sin(la))]
                )
                * radius,
            )
        )
    p = np.radians(np.linspace(-90, 90, 181))
    for lon in np.arange(-180, 180, step_deg):
        lo = np.radians(lon)
        lines.append(
            (
                lon == 0,
                np.column_stack([np.cos(p) * np.cos(lo), np.cos(p) * np.sin(lo), np.sin(p)])
                * radius,
            )
        )
    return lines


def make_widget(parent=None):
    """Qt 위젯 생성 (OpenGL을 못 쓰는 환경이면 안내 라벨)."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    try:
        import pyqtgraph.opengl as gl
        from pyqtgraph import Vector
    except Exception as exc:  # pyqtgraph/PyOpenGL 미설치
        lbl = QLabel(f"3D 화면을 쓸 수 없습니다: {exc}\n`uv sync --extra viewer --inexact`")
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    SPEEDS = [
        ("×60 (1분/초)", 60),
        ("×600", 600),
        ("×3,600 (1시간/초)", 3600),
        ("×86,400 (1일/초)", 86400),
        ("×604,800 (1주/초)", 604800),
        ("×2,592,000 (30일/초)", 2592000),
    ]
    FPS = 30

    class Orbit3D(QWidget):
        def __init__(self):
            super().__init__(parent)
            self.track: OrbitTrack | None = None
            self.curves: pd.DataFrame | None = None
            self.t = 0.0  # 시작 후 일

            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor((8, 10, 20))
            self.view.setCameraPosition(pos=Vector(0, 0, 0), distance=4.2, elevation=22, azimuth=35)

            md = gl.MeshData.sphere(rows=72, cols=144, radius=1.0)
            md.setVertexColors(earth_vertex_colors(md.vertexes()))
            self.earth = gl.GLMeshItem(
                meshdata=md, smooth=True, shader="shaded", glOptions="opaque"
            )
            self.view.addItem(self.earth)
            self.grid_items = []
            for main, pts in graticule():
                item = gl.GLLinePlotItem(
                    pos=pts,
                    width=1.5 if main else 1.0,
                    antialias=True,
                    color=(1, 1, 1, 0.55) if main else (1, 1, 1, 0.18),
                )
                self.view.addItem(item)
                self.grid_items.append(item)
            axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, -1.35], [0, 0, 1.35]]), color=(0.7, 0.7, 0.7, 0.6), width=1
            )
            self.view.addItem(axis)
            self.grid_items.append(axis)

            self.trail = gl.GLLinePlotItem(color=(1.0, 0.55, 0.1, 0.95), width=2, antialias=True)
            self.ground = gl.GLLinePlotItem(color=(1.0, 0.9, 0.2, 0.9), width=1.5, antialias=True)
            self.sat = gl.GLScatterPlotItem(color=(1, 1, 1, 1), size=11, pxMode=True)
            self.nadir = gl.GLLinePlotItem(color=(1, 1, 1, 0.35), width=1)
            for it in (self.trail, self.ground, self.nadir, self.sat):
                self.view.addItem(it)

            # 조작부
            self.play = QPushButton("▶ 재생")
            self.play.setCheckable(True)
            self.play.toggled.connect(self._toggle)
            self.speed = QComboBox()
            for label, _ in SPEEDS:
                self.speed.addItem(label)
            self.speed.setCurrentIndex(2)
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 10000)
            self.slider.valueChanged.connect(self._seek)
            self.frame = QComboBox()
            self.frame.addItems(["ECI (관성)", "ECEF (지구 고정)"])
            self.frame.currentIndexChanged.connect(self.redraw)
            self.model = QComboBox()
            self.model.currentIndexChanged.connect(self._model_changed)
            self.exag = QDoubleSpinBox()
            self.exag.setRange(1, 50)
            self.exag.setValue(1)
            self.exag.setPrefix("고도 ×")
            self.exag.valueChanged.connect(self.redraw)
            self.orbits = QDoubleSpinBox()
            self.orbits.setRange(0.2, 50)
            self.orbits.setValue(1.0)
            self.orbits.setSuffix(" 바퀴")
            self.orbits.valueChanged.connect(self.redraw)
            self.show_ground = QCheckBox("지상궤적")
            self.show_ground.setChecked(True)
            self.show_ground.toggled.connect(self.redraw)
            self.info = QLabel()
            self.info.setStyleSheet("font-family: monospace")

            row1 = QHBoxLayout()
            row1.addWidget(self.play)
            row1.addWidget(self.speed)
            row1.addWidget(self.slider, 1)
            row2 = QHBoxLayout()
            for w_ in (
                QLabel("좌표계"),
                self.frame,
                QLabel("고도 곡선"),
                self.model,
                self.exag,
                QLabel("궤적 꼬리"),
                self.orbits,
                self.show_ground,
            ):
                row2.addWidget(w_)
            row2.addStretch()
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.view, 1)
            lay.addWidget(self.info)
            lay.addLayout(row1)
            lay.addLayout(row2)

            self.timer = QTimer(self)
            self.timer.setInterval(int(1000 / FPS))
            self.timer.timeout.connect(self._tick)

        # --- 데이터 ---
        def set_scenario(self, alt0_km, inc_deg, start, curves: pd.DataFrame | None = None):
            self.curves = curves
            self._scenario = (alt0_km, inc_deg, start)
            self.model.blockSignals(True)
            self.model.clear()
            if curves is not None and not curves.empty:
                from orbital_decay.figures import model_label

                for m in dict.fromkeys(curves["model"]):
                    self.model.addItem(model_label(m), m)
            else:
                self.model.addItem("일정 고도 (시뮬레이션 전)", None)
            self.model.blockSignals(False)
            self._model_changed()

        def _model_changed(self):
            alt0, inc, start = self._scenario
            m = self.model.currentData()
            curve = None
            if self.curves is not None and m is not None:
                curve = self.curves[self.curves["model"] == m]
            self.track = OrbitTrack(alt0, inc, start, curve)
            self.t = min(self.t, self.track.duration_days)
            self._sync_slider()
            self.redraw()

        # --- 시간 ---
        def _toggle(self, on):
            self.play.setText("⏸ 정지" if on else "▶ 재생")
            self.timer.start() if on else self.timer.stop()

        def _tick(self):
            if self.track is None:
                return
            step = SPEEDS[self.speed.currentIndex()][1] / FPS / 86400
            self.t += step
            if self.t >= self.track.duration_days:
                self.t = self.track.duration_days
                self.play.setChecked(False)
            self._sync_slider()
            self.redraw()

        def _sync_slider(self):
            if self.track and self.track.duration_days > 0:
                self.slider.blockSignals(True)
                self.slider.setValue(int(self.t / self.track.duration_days * 10000))
                self.slider.blockSignals(False)

        def _seek(self, v):
            if self.track:
                self.t = v / 10000 * self.track.duration_days
                self.redraw()

        # --- 그리기 ---
        def redraw(self):
            tr = self.track
            if tr is None:
                return
            frame = "ECEF" if self.frame.currentIndex() == 1 else "ECI"
            ex = self.exag.value()
            now = tr.start + pd.Timedelta(days=self.t)
            gmst = float(np.degrees(gmst_rad(pd.DatetimeIndex([now]))[0]))
            for item in (self.earth, *self.grid_items):
                item.resetTransform()
                if frame == "ECI":
                    item.rotate(gmst, 0, 0, 1)
            pos, alt = tr.positions(np.array([self.t]), frame, ex)
            self.sat.setData(pos=pos)
            self.nadir.setData(pos=np.vstack([pos[0] / np.linalg.norm(pos[0]), pos[0]]))
            self.trail.setData(pos=tr.trail(self.t, self.orbits.value(), frame, ex))
            self.ground.setVisible(self.show_ground.isChecked())
            if self.show_ground.isChecked():
                self.ground.setData(pos=tr.ground_track(self.t, self.orbits.value(), frame))
            self.info.setText(
                f"{now:%Y-%m-%d %H:%M} UTC   경과 {self.t:8.2f}일   평균고도 {alt[0]:6.1f} km"
                f"   주기 {period_minutes(alt[0]):5.1f}분"
                + ("   (재진입)" if alt[0] <= 120.5 else "")
            )

    return Orbit3D()
