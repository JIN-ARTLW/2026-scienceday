"""ML 보정모델 후보를 위성 단위 중첩 교차검증으로 비교하고 근거와 함께 하나를 고른다.

run_models.py가 만든 dataset.parquet을 입력으로 쓴다:
    .venv/bin/python scripts/select_model.py data/processed/models/<run>/dataset.parquet
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from orbital_decay.selection import EXCLUDED, available_candidates, nested_loso, select_model


def _markdown(table: pd.DataFrame) -> str:
    head = ["model", *table.columns]
    rows = [
        [str(i), *(f"{v:.3g}" if isinstance(v, float) else str(v) for v in r)]
        for i, r in zip(table.index, table.itertuples(index=False), strict=True)
    ]
    return "\n".join("| " + " | ".join(r) + " |" for r in [head, ["---"] * len(head), *rows])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", help="run_models.py 결과의 dataset.parquet")
    p.add_argument("--only", nargs="*", help="이 후보만 비교 (이름)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", module="sklearn.gaussian_process")

    dataset = Path(args.dataset)
    df = pd.read_parquet(dataset)
    out = dataset.parent / "selection"
    out.mkdir(exist_ok=True)

    candidates = available_candidates()
    if args.only:
        candidates = [c for c in candidates if c.name in args.only]
    print(
        f"위성 {df['norad_id'].nunique()}기, 후보 {len(candidates)}개: "
        f"{', '.join(c.name for c in candidates)}"
    )

    results = nested_loso(df, candidates, n_jobs=args.n_jobs)
    results.to_csv(out / "per_satellite.csv", index=False)
    table, chosen, reasons = select_model(results, candidates, alpha=args.alpha)
    table.to_csv(out / "summary.csv")
    params = (
        results[results["params"] != ""]
        .groupby(["model", "params"])
        .size()
        .rename("folds")
        .reset_index()
    )
    params.to_csv(out / "chosen_params.csv", index=False)

    cols = [
        "family",
        "mean_skill",
        "skill_ci_low",
        "skill_ci_high",
        "sats_better_than_physics",
        "sats_better_than_fixed_k",
        "p_vs_best",
        "p_better_than_fixed_k",
        "fit_seconds",
    ]
    print(table[cols].round(3).to_string())
    print("\n선택:", chosen)
    for r in reasons:
        print(" -", r)

    lines = [
        "# ML 보정모델 선택 보고서",
        "",
        f"자료: `{dataset}` (위성 {df['norad_id'].nunique()}기)",
        "",
        "## 방법",
        "",
        "- 모든 후보: 같은 특징, 같은 목표(배율 = 관측/물리), 같은 가중치(물리²)",
        "  → 모두 감쇠율 제곱오차 최소화",
        "- 바깥 루프: 위성 하나씩 빼기(LOSO)",
        "- 안쪽 루프: 학습 위성만으로 GroupKFold 하이퍼파라미터 선택",
        "- skill = 1 − RMSE_모델 / RMSE_순수물리 (위성별), 95% CI는 위성 부트스트랩",
        f"- 선택 규칙: 최고 후보와 Wilcoxon p ≥ {args.alpha}인 후보 중 가장 단순한 모델",
        "",
        "## 결과",
        "",
        _markdown(table[cols]),
        "",
        f"## 선택: `{chosen}`",
        "",
        *[f"- {r}" for r in reasons],
        "",
        "## 후보에서 제외한 방법",
        "",
        *[f"- **{k}**: {v}" for k, v in EXCLUDED.items()],
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
