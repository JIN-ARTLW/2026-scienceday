"""결과 뷰어 (데스크톱 창).

    uv run python scripts/viewer.py                 # 창 열기
    uv run python scripts/viewer.py --export        # 창 없이 최신 결과 그림을 PNG로 저장

결과 폴더(<데이터 폴더>/processed/models/)를 읽기만 하므로, 계산은 다른 컴퓨터에서 하고
Google Drive로 동기화된 결과를 어느 컴퓨터에서든 볼 수 있다.
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path

import pandas as pd

from orbital_decay import figures, orbit3d
from orbital_decay.results import MODELS_DIR, Run, list_runs


def export(run_dir: Path | None, models_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    runs = [Path(run_dir)] if run_dir else list_runs(models_dir)[:1]
    if not runs:
        sys.exit(f"결과 폴더가 없습니다: {models_dir}")
    run = Run(runs[0])
    saved = figures.export_all(run)
    print(f"{run.name}: 그림 {len(saved)}개 저장 → {saved[0].parent}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="궤도 감쇠 모델 결과 뷰어")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="결과 폴더들의 상위 폴더")
    p.add_argument("--run", type=Path, help="특정 결과 폴더")
    p.add_argument("--export", action="store_true", help="창 없이 PNG만 저장")
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore", message="There is data that was either interpolated")
    if args.export:
        export(args.run, args.models_dir)
        return
    _run_gui(args.models_dir, args.run)


def _run_gui(models_dir: Path, run_dir: Path | None) -> None:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    from PySide6.QtCore import QObject, Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDateEdit,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )

    from orbital_decay.simulate import Scenario, simulate, space_weather_end

    class Plot(QWidget):
        """matplotlib 캔버스 + 확대/이동 도구 막대."""

        def __init__(self, size=(9, 6)):
            super().__init__()
            self.figure = Figure(figsize=size)
            self.canvas = FigureCanvasQTAgg(self.figure)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(NavigationToolbar2QT(self.canvas, self))
            lay.addWidget(self.canvas)

        def draw(self, fn, *args):
            try:
                fn(self.figure, *args)
            except Exception as exc:
                self.figure.clear()
                figures._empty(self.figure, f"그리기 실패: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            self.canvas.draw_idle()

    def fill_table(table: QTableWidget, df: pd.DataFrame | None, digits: int = 3):
        table.clear()
        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        if not isinstance(df.index, pd.RangeIndex) or df.index.name:
            df = df.reset_index()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i, row in enumerate(df.itertuples(index=False)):
            for j, v in enumerate(row):
                text = f"{v:.{digits}g}" if isinstance(v, float) else str(v)
                table.setItem(i, j, QTableWidgetItem(text))
        table.resizeColumnsToContents()

    def sat_list_panel(on_select):
        lst = QListWidget()
        lst.currentTextChanged.connect(lambda t: t and on_select(int(t.split()[0])))
        return lst

    class SimWorker(QObject):
        done = Signal(object, object, str)
        failed = Signal(str)

        def __init__(self, scenario, models, title):
            super().__init__()
            self.scenario, self.models, self.title = scenario, models, title

        def run(self):
            try:
                curves, info = simulate(self.scenario, self.models)
                self.done.emit(curves, info, self.title)
            except Exception as exc:
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    class Viewer(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("궤도 감쇠 모델 결과 뷰어")
            self.resize(1400, 900)
            self.models_dir = models_dir
            self.run: Run | None = None
            self._thread = None

            top = QHBoxLayout()
            self.dir_label = QLabel()
            self.dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.run_box = QComboBox()
            self.run_box.setMinimumWidth(320)
            self.run_box.currentIndexChanged.connect(self.load_run)
            btn_dir = QPushButton("폴더 선택")
            btn_dir.clicked.connect(self.choose_dir)
            btn_refresh = QPushButton("새로고침")
            btn_refresh.clicked.connect(self.refresh_runs)
            btn_export = QPushButton("PNG 내보내기")
            btn_export.clicked.connect(self.export_png)
            top.addWidget(QLabel("결과:"))
            top.addWidget(self.run_box)
            top.addWidget(btn_refresh)
            top.addWidget(btn_export)
            top.addStretch()
            top.addWidget(self.dir_label)
            top.addWidget(btn_dir)

            self.tabs = QTabWidget()
            self.tabs.addTab(self._summary_tab(), "요약")
            self.tabs.addTab(self._satellite_tab(), "위성별")
            self.tabs.addTab(self._propagation_tab(), "궤도 전파")
            self.tabs.addTab(self._solar_tab(), "태양활동")
            self.tabs.addTab(self._simulation_tab(), "시뮬레이션")

            central = QWidget()
            lay = QVBoxLayout(central)
            lay.addLayout(top)
            lay.addWidget(self.tabs)
            self.setCentralWidget(central)
            self.refresh_runs(select=run_dir)

        # ---------- 탭 구성 ----------
        def _summary_tab(self):
            w = QSplitter(Qt.Horizontal)
            left = QWidget()
            ll = QVBoxLayout(left)
            self.info_text = QTextBrowser()
            self.summary_table = QTableWidget()
            ll.addWidget(QLabel("실행 정보 · 모델 선택 근거"))
            ll.addWidget(self.info_text, 3)
            ll.addWidget(QLabel("감쇠율 오차 (m/day)"))
            ll.addWidget(self.summary_table, 2)
            self.summary_plot = Plot()
            w.addWidget(left)
            w.addWidget(self.summary_plot)
            w.setSizes([550, 850])
            return w

        def _satellite_tab(self):
            w = QSplitter(Qt.Horizontal)
            left = QWidget()
            ll = QVBoxLayout(left)
            self.sat_list = sat_list_panel(self.show_satellite)
            self.sat_table = QTableWidget()
            self.pred_source = QLabel()
            ll.addWidget(QLabel("위성 (NORAD  평균고도  BC)"))
            ll.addWidget(self.sat_list, 3)
            ll.addWidget(self.pred_source)
            ll.addWidget(self.sat_table, 2)
            self.sat_plot = Plot((10, 8))
            w.addWidget(left)
            w.addWidget(self.sat_plot)
            w.setSizes([380, 1020])
            return w

        def _propagation_tab(self):
            w = QSplitter(Qt.Horizontal)
            left = QWidget()
            ll = QVBoxLayout(left)
            self.prop_list = sat_list_panel(self.show_propagation)
            self.prop_table = QTableWidget()
            ll.addWidget(QLabel("검증 위성"))
            ll.addWidget(self.prop_list, 2)
            ll.addWidget(QLabel("전파 고도 오차 [km] (예측 − 관측)"))
            ll.addWidget(self.prop_table, 2)
            self.prop_plot = Plot()
            w.addWidget(left)
            w.addWidget(self.prop_plot)
            w.setSizes([380, 1020])
            return w

        def _solar_tab(self):
            self.solar_plot = Plot((12, 5))
            return self.solar_plot

        def _simulation_tab(self):
            w = QSplitter(Qt.Horizontal)
            box = QGroupBox("시나리오 (원궤도)")
            form = QFormLayout(box)

            def spin(lo, hi, val, step, dec=1, suffix=""):
                s = QDoubleSpinBox()
                s.setRange(lo, hi)
                s.setValue(val)
                s.setSingleStep(step)
                s.setDecimals(dec)
                s.setSuffix(suffix)
                return s

            self.sim_alt = spin(200, 900, 450, 10, 0, " km")
            self.sim_inc = spin(0, 180, 51.6, 1, 1, "°")
            self.sim_mass = spin(0.1, 20000, 4.0, 0.5, 2, " kg")
            self.sim_area = spin(0.001, 500, 0.03, 0.005, 4, " m²")
            self.sim_cd = spin(1.0, 4.0, 2.2, 0.1, 2)
            self.sim_bc = QLabel()
            for s in (self.sim_mass, self.sim_area, self.sim_cd):
                s.valueChanged.connect(self._update_bc)
            self.sim_start = QDateEdit()
            self.sim_start.setCalendarPopup(True)
            self.sim_start.setDisplayFormat("yyyy-MM-dd")
            self.sim_start.setDate(pd.Timestamp("2014-01-01").date())
            self.sim_days = QSpinBox()
            self.sim_days.setRange(30, 36500)
            self.sim_days.setValue(3650)
            self.sim_days.setSuffix(" 일")
            self.sim_scale = spin(0.3, 2.0, 1.0, 0.1, 2, " ×")
            self.sim_models = QLabel()
            self.sim_models.setWordWrap(True)
            self.sim_button = QPushButton("시뮬레이션 실행")
            self.sim_button.clicked.connect(self.run_simulation)
            self.sim_status = QLabel()
            self.sim_status.setWordWrap(True)
            form.addRow("초기 고도", self.sim_alt)
            form.addRow("경사각", self.sim_inc)
            form.addRow("질량", self.sim_mass)
            form.addRow("단면적", self.sim_area)
            form.addRow("항력계수 C_D", self.sim_cd)
            form.addRow("BC = C_D·A/m", self.sim_bc)
            form.addRow("시작일", self.sim_start)
            form.addRow("기간", self.sim_days)
            form.addRow("태양활동 배율", self.sim_scale)
            form.addRow("사용 모델", self.sim_models)
            form.addRow(self.sim_button)
            form.addRow(self.sim_status)
            note = QLabel(
                "3U 큐브위성 예: 4 kg, 0.03 m² (평균 단면적)\n"
                f"태양·지자기 자료 끝: {space_weather_end():%Y-%m-%d}\n"
                "그 이후는 과거 주기(SC24)를 다시 쓰는 시나리오입니다."
            )
            note.setStyleSheet("color: #6b7280")
            form.addRow(note)
            self._update_bc()

            right = QTabWidget()
            self.sim_plot = Plot()
            self.sim_plot.draw(
                figures._empty, "왼쪽에서 조건을 정하고 [시뮬레이션 실행]을 누르세요 (약 1~3초)"
            )
            self.orbit3d = orbit3d.make_widget()
            self.shape_plot = Plot((12, 5))
            right.addTab(self.sim_plot, "고도 감쇠")
            right.addTab(self.orbit3d, "3D 궤도")
            right.addTab(self.shape_plot, "궤도 모양 · 지상궤적")
            for s_ in (self.sim_alt, self.sim_inc):
                s_.valueChanged.connect(self.show_shape)
            self.sim_start.dateChanged.connect(self.show_shape)
            w.addWidget(box)
            w.addWidget(right)
            w.setSizes([380, 1020])
            self.show_shape()
            return w

        # ---------- 동작 ----------
        def choose_dir(self):
            d = QFileDialog.getExistingDirectory(
                self, "결과 폴더들의 상위 폴더", str(self.models_dir)
            )
            if d:
                self.models_dir = Path(d)
                self.refresh_runs()

        def refresh_runs(self, select=None):
            self.dir_label.setText(str(self.models_dir))
            runs = list_runs(self.models_dir)
            current = select or (self.run.path if self.run else None)
            self.run_box.blockSignals(True)
            self.run_box.clear()
            for r in runs:
                self.run_box.addItem(r.name, str(r))
            self.run_box.blockSignals(False)
            if not runs:
                self.info_text.setMarkdown(
                    f"결과 폴더가 없습니다.\n\n`{self.models_dir}`\n\n"
                    "`scripts/run_models.py`를 먼저 실행하거나 **폴더 선택**으로 위치를 지정하세요."
                )
                return
            idx = next((i for i, r in enumerate(runs) if current and r == Path(current)), 0)
            self.run_box.setCurrentIndex(idx)
            self.load_run()

        def load_run(self):
            path = self.run_box.currentData()
            if not path:
                return
            try:
                self.run = Run(Path(path))
            except Exception as exc:
                QMessageBox.warning(self, "불러오기 실패", str(exc))
                return
            run = self.run
            info = run.info
            ds = run.dataset
            models, model_err = run.models()
            _, report = run.selection()
            lines = [
                f"### {run.name}",
                f"- 위성 {ds['norad_id'].nunique()}기, 감쇠율 창 {len(ds)}개",
                f"- 학습 위성: {info.get('train_ids')}",
                f"- 검증 위성: {info.get('test_ids')}",
                f"- 고정 보정계수 k = {info.get('fixed_k', float('nan')):.3f}",
                f"- ML 모델: {info.get('ml_backend', info.get('ml_model'))}",
                f"- 예측 표시: {run.prediction_source}",
            ]
            if model_err:
                lines.append(f"- ⚠ 시뮬레이션용 모델: {model_err}")
            if info.get("args", {}).get("synthetic"):
                lines.append("- ⚠ **가상 위성 자료** (파이프라인 확인용, 연구 결과 아님)")
            self.info_text.setMarkdown(
                "\n".join(lines) + ("\n\n---\n\n" + report if report else "")
            )
            summary = run.table("summary_loso")
            fill_table(
                self.summary_table, (summary if summary is not None else run.table("summary_test"))
            )
            self.summary_plot.draw(figures.model_summary, run)

            sats = ds.groupby("norad_id").agg(alt=("alt_mid_km", "median"), bc=("bc_est", "first"))
            self.sat_list.blockSignals(True)
            self.sat_list.clear()
            for nid, r in sats.iterrows():
                self.sat_list.addItem(f"{nid}   {r.alt:6.0f} km   {r.bc:.4f}")
            self.sat_list.blockSignals(False)
            self.pred_source.setText(f"예측: {run.prediction_source}")
            curves = run.table("propagation_curves")
            self.prop_list.blockSignals(True)
            self.prop_list.clear()
            if curves is not None:
                for nid in sorted(curves["norad_id"].unique()):
                    self.prop_list.addItem(str(nid))
            self.prop_list.blockSignals(False)
            if self.sat_list.count():
                self.sat_list.setCurrentRow(0)
            if self.prop_list.count():
                self.prop_list.setCurrentRow(0)
            else:
                self.prop_plot.draw(figures._empty, "전파 결과가 없습니다")
            self.solar_plot.draw(figures.solar, run)
            self.sim_models.setText(
                ", ".join(["physics", *models.keys() - {"physics"}]) or "physics"
            )

        def show_satellite(self, norad_id):
            if self.run:
                self.sat_plot.draw(figures.satellite, self.run, norad_id)
                fill_table(self.sat_table, figures.satellite_metrics(self.run, norad_id))

        def show_propagation(self, norad_id):
            if not self.run:
                return
            self.prop_plot.draw(figures.propagation, self.run, norad_id)
            err = self.run.table("propagation_errors")
            if err is not None:
                e = err[err["norad_id"] == norad_id].pivot_table(
                    index="horizon_days", columns="model", values="alt_error_km"
                )
                fill_table(self.prop_table, e)

        def _update_bc(self):
            bc = Scenario.bc_from(self.sim_mass.value(), self.sim_area.value(), self.sim_cd.value())
            self.sim_bc.setText(f"{bc:.4f} m²/kg")
            return bc

        def show_shape(self):
            """입력이 바뀌면 2D 모양과 3D 궤도를 (감쇠 없이) 새 조건으로 다시 그린다."""
            self.shape_plot.draw(figures.orbit_shape, self.sim_alt.value(), self.sim_inc.value())
            if hasattr(self.orbit3d, "set_scenario"):
                self.orbit3d.set_scenario(
                    self.sim_alt.value(),
                    self.sim_inc.value(),
                    self.sim_start.date().toString("yyyy-MM-dd"),
                )

        def run_simulation(self):
            sc = Scenario(
                alt0_km=self.sim_alt.value(),
                inc_deg=self.sim_inc.value(),
                bc=self._update_bc(),
                start=self.sim_start.date().toString("yyyy-MM-dd"),
                days=self.sim_days.value(),
                f107_scale=self.sim_scale.value(),
                area_to_mass=self.sim_area.value() / self.sim_mass.value(),
            )
            models = {"physics": None}
            if self.run:
                loaded, _ = self.run.models()
                models.update({k: v for k, v in loaded.items() if k != "physics"})
            title = (
                f"{sc.alt0_km:.0f} km, 경사각 {sc.inc_deg:.1f}°, BC {sc.bc:.4f} m²/kg, "
                f"시작 {sc.start}, 태양활동 ×{sc.f107_scale:.2f}"
            )
            self.sim_button.setEnabled(False)
            self.sim_status.setText("계산 중… (MSIS 밀도 격자 → 궤도 전파)")
            self._thread = QThread()
            self._worker = SimWorker(sc, models, title)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.done.connect(self._sim_done)
            self._worker.failed.connect(self._sim_failed)
            self._worker.done.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._thread.start()

        def _sim_done(self, curves, info, title):
            self.sim_plot.draw(figures.simulation, curves, info, title)
            if hasattr(self.orbit3d, "set_scenario"):
                sc = self._worker.scenario
                self.orbit3d.set_scenario(sc.alt0_km, sc.inc_deg, sc.start, curves)
            life = ", ".join(
                f"{figures.model_label(k)}: {v / 365.25:.1f}년"
                if v
                else f"{figures.model_label(k)}: 기간 내 재진입 없음"
                for k, v in info["lifetime_days"].items()
            )
            self.sim_status.setText(f"완료. {life}")
            self.sim_button.setEnabled(True)

        def _sim_failed(self, message):
            self.sim_status.setText(f"실패: {message}")
            self.sim_button.setEnabled(True)

        def export_png(self):
            if not self.run:
                return
            saved = figures.export_all(self.run)
            QMessageBox.information(self, "PNG 저장", f"{len(saved)}개 저장:\n{saved[0].parent}")

    app = QApplication.instance() or QApplication(sys.argv)
    win = Viewer()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
