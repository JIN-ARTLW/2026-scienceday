"""물리모델 / 고정 보정계수 모델 / ML 보정모델 학습·비교 실행.

실제 자료 (Orbitoby에서 내보낸 orbit_elements):
    .venv/bin/python scripts/run_models.py --elements data/processed/orbit_elements.parquet

Orbitoby DuckDB를 바로 읽기 (duckdb 설치 필요):
    .venv/bin/python scripts/run_models.py \
        --orbitoby-db ~/.orbitoby/warehouse/archive.duckdb --norad 22 29 45

가상 자료로 파이프라인 확인:
    .venv/bin/python scripts/run_models.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from orbital_decay.analysis import lag_correlation
from orbital_decay.config import PROCESSED_DIR
from orbital_decay.dataset import build_dataset
from orbital_decay.density import DensityLookup, density_table
from orbital_decay.evaluation import (
    activity_bins,
    compare_models,
    leave_one_satellite_out,
    propagation_errors,
    split_satellites,
    summarize,
)
from orbital_decay.models import (
    FixedCorrectionModel,
    MLCorrectionModel,
    PhysicsModel,
    feature_importance,
    freeze,
)
from orbital_decay.orbit import decay_windows, prepare_history
from orbital_decay.spaceweather import daily_space_weather


def load_elements(args) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if args.synthetic:
        from orbital_decay.synthetic import synthetic_fleet

        elements, truth, meta = synthetic_fleet(
            n_satellites=args.synthetic_n, n_days=args.synthetic_days, seed=args.seed
        )
        return elements, meta
    if args.orbitoby_db:
        import duckdb

        con = duckdb.connect(str(Path(args.orbitoby_db).expanduser()), read_only=True)
        query = "SELECT * FROM orbit_elements"
        if args.norad:
            query += f" WHERE norad_id IN ({','.join(str(int(n)) for n in args.norad)})"
        elements = con.execute(query).df()
    else:
        path = Path(args.elements)
        elements = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if args.norad:
            elements = elements[elements["norad_id"].isin(args.norad)]
    meta = pd.read_csv(args.metadata) if args.metadata else None
    return elements, meta


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--elements", help="orbit_elements parquet/csv (Orbitoby 열 이름)")
    src.add_argument("--orbitoby-db", help="Orbitoby DuckDB 파일 경로")
    src.add_argument("--synthetic", action="store_true", help="가상 위성으로 실행")
    p.add_argument("--norad", type=int, nargs="*", help="사용할 NORAD ID")
    p.add_argument("--metadata", help="norad_id,mass_kg,area_m2 CSV (선택)")
    p.add_argument("--calib-days", type=float, default=365, help="BC 추정 구간 길이 [일]")
    p.add_argument("--window-days", type=float, default=14, help="감쇠율 창 길이 [일]")
    p.add_argument(
        "--min-alt-km", type=float, default=250, help="이보다 낮은 창(재진입 직전)은 제외"
    )
    p.add_argument("--test-fraction", type=float, default=1 / 3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loso", action="store_true", help="위성 하나씩 빼는 교차검증도 실행")
    p.add_argument("--backend", default="auto", choices=["auto", "xgboost", "sklearn"])
    p.add_argument(
        "--ml-model",
        help="select_model.py가 고른 후보 이름 (예: ridge, xgboost). 없으면 기본 부스팅 모델",
    )
    p.add_argument("--synthetic-n", type=int, default=9)
    p.add_argument("--synthetic-days", type=int, default=1500)
    p.add_argument("--out", help="결과 폴더 (기본: <데이터 폴더>/processed/models/<시각>_<컴퓨터>)")
    args = p.parse_args()

    warnings.filterwarnings("ignore", message="There is data that was either interpolated")

    out = (
        Path(args.out)
        if args.out
        # 공유 폴더(Google Drive)에서 두 컴퓨터의 결과가 겹치지 않도록 컴퓨터 이름을 붙인다
        else PROCESSED_DIR
        / "models"
        / f"{datetime.now():%Y%m%d_%H%M%S}_{platform.node().split('.')[0]}"
    )
    out.mkdir(parents=True, exist_ok=True)

    elements, meta = load_elements(args)
    history = prepare_history(elements)
    windows = decay_windows(history, window_days=args.window_days)
    print(
        f"위성 {history['norad_id'].nunique()}기, TLE {len(history)}개, 감쇠율 창 {len(windows)}개"
    )

    table = density_table(history, cache_dir=PROCESSED_DIR / "density_cache")
    lookup = DensityLookup(table)
    sw = daily_space_weather(history["epoch"].min(), history["epoch"].max())
    df = build_dataset(
        windows, lookup, sw, calib_days=args.calib_days, metadata=meta, min_alt_km=args.min_alt_km
    )
    print(f"재진입 직전 저고도 창 제외: {df.attrs['excluded_low_alt_windows']}개")
    if df.attrs.get("dropped_no_bc"):
        print("BC 추정 불가로 제외된 위성:", df.attrs["dropped_no_bc"])

    if args.ml_model:
        from orbital_decay.selection import CandidateCorrection, available_candidates

        by_name = {c.name: c for c in available_candidates(args.seed)}
        if args.ml_model not in by_name:
            p.error(f"--ml-model: 이 컴퓨터에서 쓸 수 있는 후보는 {sorted(by_name)}")
        ml_model = CandidateCorrection(by_name[args.ml_model])
    else:
        ml_model = MLCorrectionModel(backend=args.backend)
    models = [PhysicsModel(), FixedCorrectionModel(), ml_model]
    train_ids, test_ids = split_satellites(df["norad_id"].unique(), args.test_fraction, args.seed)
    summary, pred, fitted = compare_models(df, train_ids, test_ids, models)
    pred["activity"] = activity_bins(pred)

    ml = fitted[-1]
    print(f"\n학습 위성 {train_ids}\n검증 위성 {test_ids}")
    ml_label = getattr(ml, "backend_", ml.name)
    print(f"고정 보정계수 k = {fitted[1].k_:.3f},  ML 모델 = {ml_label}")
    print("\n[검증 위성 감쇠율 오차]")
    print(summary.round(3).to_string())
    by_activity = summarize(pred, fitted, by="activity")
    print("\n[태양활동 조건별]")
    print(by_activity[["n", "rmse_m_day", "median_abs_pct"]].round(3).to_string())

    prop, curves = propagation_errors(df, test_ids, fitted, sw, lookup)
    if not prop.empty:
        prop_summary = prop.assign(abs_km=prop["alt_error_km"].abs()).pivot_table(
            index="horizon_days", columns="model", values="abs_km", aggfunc="median"
        )
        print("\n[궤도 전파 고도 오차 중앙값 |km|]")
        print(prop_summary.round(3).to_string())
        prop.to_csv(out / "propagation_errors.csv", index=False)
        curves.to_parquet(out / "propagation_curves.parquet", index=False)

    importance = feature_importance(ml, pred.assign(is_calib=False), random_state=args.seed)
    print("\n[ML 특징 중요도 (검증 위성)]")
    print(importance.round(4).to_string())

    lag = lag_correlation(df, sw, "f107_obs", 30)
    best = lag.loc[lag["r"].idxmax()]
    print(f"\nF10.7 지연 상관 최대: lag {int(best.lag_days)}일, r={best.r:.3f}")

    summary.to_csv(out / "summary_test.csv")
    by_activity.to_csv(out / "summary_by_activity.csv")
    summarize(pred, fitted, by="norad_id").to_csv(out / "summary_by_satellite.csv")
    pred.to_parquet(out / "predictions_test.parquet", index=False)
    df.to_parquet(out / "dataset.parquet", index=False)
    importance.to_csv(out / "feature_importance.csv")
    lag.to_csv(out / "lag_correlation_f107.csv", index=False)
    # 뷰어·시뮬레이션용: 학습 위성으로 학습한 모델 (예측에 필요한 부분만 저장)
    joblib.dump({m.name: freeze(m) for m in fitted}, out / "models.joblib")
    if meta is not None:
        meta.to_csv(out / "metadata.csv", index=False)

    if args.loso:
        loso = leave_one_satellite_out(df, models)
        loso_summary = summarize(loso, models)
        print("\n[위성 하나씩 빼는 교차검증]")
        print(loso_summary.round(3).to_string())
        loso_summary.to_csv(out / "summary_loso.csv")
        loso.to_parquet(out / "predictions_loso.parquet", index=False)

    (out / "run.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "train_ids": train_ids,
                "test_ids": test_ids,
                "fixed_k": fitted[1].k_,
                "ml_model": ml.name,
                "ml_backend": ml_label,
                "features": ml.features,
                "cleaning": {str(k): v for k, v in history.attrs.get("cleaning", {}).items()},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
