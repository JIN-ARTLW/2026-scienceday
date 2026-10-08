"""Ashruf et al. (2026) Table 2(태양주기 극대기 감쇠율) 재현과 비교.

    .venv/bin/python scripts/reproduce_ashruf.py                 # data/raw/orbit_elements.parquet
    .venv/bin/python scripts/reproduce_ashruf.py --elements <parquet>

결과: <데이터 폴더>/processed/ashruf_reproduction/
  table2_comparison.csv, windows.csv, table2_comparison.png, report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from orbital_decay import figures  # noqa: F401  (한글 글꼴 설정)
from orbital_decay.ashruf import (
    PAPER_TABLE2,
    apparent_altitude,
    peak_decay_rates,
    rapid_windows,
    smoothed_rate,
)
from orbital_decay.config import PROCESSED_DIR, RAW_DIR


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--elements", default=str(RAW_DIR / "orbit_elements.parquet"))
    p.add_argument("--method", default="span", choices=["span", "longest"])
    p.add_argument("--out", default=str(PROCESSED_DIR / "ashruf_reproduction"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    alt = apparent_altitude(pd.read_parquet(args.elements))
    per, common = rapid_windows(smoothed_rate(alt), args.method)
    ours = peak_decay_rates(alt, common)
    ids = ours.index.intersection(PAPER_TABLE2.index)
    rows = []
    for cyc in ("SC22", "SC23", "SC24"):
        for nid in ids:
            rows.append(
                {
                    "norad_id": nid,
                    "cycle": cyc,
                    "ours_m_h": ours.loc[nid, cyc],
                    "paper_m_h": PAPER_TABLE2.loc[nid, cyc],
                }
            )
    cmp_ = pd.DataFrame(rows)
    cmp_["diff_m_h"] = cmp_["ours_m_h"] - cmp_["paper_m_h"]
    cmp_.to_csv(out / "table2_comparison.csv", index=False)
    per.to_csv(out / "windows.csv", index=False)

    stats = cmp_.groupby("cycle").agg(
        n=("diff_m_h", "size"),
        mean_abs_diff=("diff_m_h", lambda d: d.abs().mean()),
        max_abs_diff=("diff_m_h", lambda d: d.abs().max()),
        r=("ours_m_h", lambda s: np.corrcoef(s, cmp_.loc[s.index, "paper_m_h"])[0, 1]),
        ours_mean=("ours_m_h", "mean"),
        paper_mean=("paper_m_h", "mean"),
    )
    print("공통 급감쇠 구간:\n", common.to_string(), "\n")
    print(stats.round(3).to_string())

    fig = Figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111)
    colors = {"SC22": "#2563eb", "SC23": "#16a34a", "SC24": "#dc2626"}
    for cyc, g in cmp_.groupby("cycle"):
        ax.scatter(g["paper_m_h"], g["ours_m_h"], color=colors[cyc], label=cyc, s=30)
    lo = min(cmp_["paper_m_h"].min(), cmp_["ours_m_h"].min()) * 1.05
    ax.plot([lo, 0], [lo, 0], color="#6b7280", lw=1, ls="--", label="y = x")
    ax.set_xlabel("논문 Table 2 [m/h]")
    ax.set_ylabel("재현 [m/h]")
    ax.set_title(f"Ashruf et al. (2026) Table 2 재현 (위성 {len(ids)}기)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "table2_comparison.png", dpi=160)

    (out / "report.md").write_text(
        "# Ashruf et al. (2026) Table 2 재현\n\n"
        f"- 자료: `{args.elements}` (Orbitoby · Space-Track GP_HISTORY)\n"
        f"- 급감쇠 구간 방법: `{args.method}` (논문은 공통 구간을 그림과 대조해 수동 조정)\n\n"
        "## 공통 급감쇠 구간\n\n```\n" + common.to_string() + "\n```\n\n"
        "## 주기별 일치도 (m/h)\n\n```\n" + stats.round(3).to_string() + "\n```\n"
    )
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
