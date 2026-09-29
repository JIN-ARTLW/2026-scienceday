"""모델 실행 결과 폴더 읽기 (뷰어·그림 내보내기 공용).

폴더 구조는 scripts/run_models.py, scripts/select_model.py가 만든다:
  <데이터 폴더>/processed/models/<시각>_<컴퓨터>/
    run.json, summary_*.csv, predictions_*.parquet, dataset.parquet,
    propagation_*.{csv,parquet}, lag_correlation_f107.csv, models.joblib,
    selection/{summary.csv, per_satellite.csv, report.md}
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from orbital_decay.config import PROCESSED_DIR

MODELS_DIR = PROCESSED_DIR / "models"


def list_runs(root: Path = MODELS_DIR) -> list[Path]:
    """결과 폴더 목록 (최신 순)."""
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if (p / "run.json").exists()), reverse=True)


class Run:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.info = json.loads((self.path / "run.json").read_text())
        self._tables: dict[str, pd.DataFrame | None] = {}
        self._models: tuple[dict, str | None] | None = None

    @property
    def name(self) -> str:
        return self.path.name

    def table(self, name: str) -> pd.DataFrame | None:
        """name: 확장자 없는 파일 이름 (예: 'summary_test', 'selection/summary'). 캐시됨."""
        if name not in self._tables:
            self._tables[name] = None
            for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
                f = self.path / f"{name}{ext}"
                if f.exists():
                    self._tables[name] = reader(f)
                    break
        return self._tables[name]

    @property
    def dataset(self) -> pd.DataFrame:
        return self.table("dataset")

    @property
    def model_names(self) -> list[str]:
        pred = self.predictions()
        if pred is None:
            return []
        return [c.removeprefix("pred_") for c in pred.columns if c.startswith("pred_")]

    def predictions(self) -> pd.DataFrame | None:
        """모든 위성의 창별 예측이 있으면 LOSO 결과, 없으면 검증 위성 결과."""
        loso = self.table("predictions_loso")
        return loso if loso is not None else self.table("predictions_test")

    @property
    def prediction_source(self) -> str:
        return "LOSO(모든 위성)" if self.table("predictions_loso") is not None else "검증 위성"

    def selection(self) -> tuple[pd.DataFrame | None, str | None]:
        summary = self.table("selection/summary")
        report = self.path / "selection" / "report.md"
        return summary, report.read_text() if report.exists() else None

    def models(self) -> tuple[dict, str | None]:
        """저장된 학습 모델. 불러오지 못하면 (빈 dict, 이유)."""
        if self._models is None:
            self._models = self._load_models()
        return self._models

    def _load_models(self) -> tuple[dict, str | None]:
        f = self.path / "models.joblib"
        if not f.exists():
            return {}, "models.joblib이 없습니다 (이전 버전 결과)."
        try:
            import joblib

            return joblib.load(f), None
        except Exception as exc:  # 예: 윈도우에서 만든 XGBoost 모델을 libomp 없는 맥에서 열 때
            return {}, f"모델을 불러오지 못했습니다: {type(exc).__name__}: {exc}"
