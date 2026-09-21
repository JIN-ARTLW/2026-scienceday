import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


DATA_DIR = _resolve_path(
    os.getenv("SCIENCEDAY_DATA_DIR", "./data")
)

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

GP_HISTORY_DIR = RAW_DIR / "gp_history"
SOLAR_DIR = RAW_DIR / "solar"

SPACETRACK_USERNAME = os.getenv("SPACETRACK_USERNAME")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD")


def ensure_data_dirs() -> None:
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        SAMPLE_DIR,
        GP_HISTORY_DIR,
        SOLAR_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
