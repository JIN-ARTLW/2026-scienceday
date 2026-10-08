"""Orbitoby로 Space-Track GP 궤도 이력을 받아 분석용 parquet 스냅샷으로 저장한다.

    # 계획만 보기 (네트워크 요청 없음)
    .venv/bin/python scripts/fetch_orbits.py --norad 228 --start 2014-01-01 --end 2015-01-01 \
        --dry-run

    # 받기 (위성 목록 CSV: norad_id 열, 선택적으로 start/end 열)
    .venv/bin/python scripts/fetch_orbits.py --satellites data/sample/satellites.csv \
        --start 1986-09-01 --end 2020-01-01

- Space-Track 계정은 이 프로젝트 .env의 SPACETRACK_USERNAME/PASSWORD를 쓴다.
- Orbitoby 로컬 DB(~/.orbitoby)에 원본·출처가 쌓이고, 이미 받은 기간은 다시 요청하지 않는다.
- 긴 기간은 --chunk-years 단위로 나눠 요청하고, 요청 사이에 --pause초 쉰다
  (Space-Track 권장 한도: 분당 30회, 시간당 300회).
- 결과: <데이터 폴더>/raw/orbit_elements.parquet (+ 같은 이름 .manifest.json)
  Space-Track 자료는 재배포 제한이 있으니 공개 저장소에 올리지 않는다 (data/raw는 git 제외).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, date, datetime
from importlib.metadata import version

import pandas as pd

from orbital_decay.config import PROJECT_ROOT, RAW_DIR


def chunks(start: date, end: date, years: int):
    cur = start
    while cur < end:
        nxt = min(date(cur.year + years, cur.month, cur.day), end)
        yield cur, nxt
        cur = nxt


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--norad", type=int, nargs="+", help="NORAD ID 목록")
    who.add_argument("--satellites", help="norad_id(필수), start/end(선택) 열이 있는 CSV")
    p.add_argument("--start", default="1986-09-01", help="기본 시작일 [포함]")
    p.add_argument("--end", default="2020-01-01", help="기본 종료일 [미포함]")
    p.add_argument("--chunk-years", type=int, default=5)
    p.add_argument("--pause", type=float, default=15.0, help="Space-Track 요청 사이 대기 [초]")
    p.add_argument("--out", default=str(RAW_DIR / "orbit_elements.parquet"))
    p.add_argument("--dry-run", action="store_true", help="요청 계획만 출력")
    args = p.parse_args()

    if args.satellites:
        sats = pd.read_csv(args.satellites)
    else:
        sats = pd.DataFrame({"norad_id": args.norad})
    for col, default in (("start", args.start), ("end", args.end)):
        values = sats[col].fillna(default) if col in sats else pd.Series(default, index=sats.index)
        sats[col] = pd.to_datetime(values).dt.date
    sats["norad_id"] = sats["norad_id"].astype(int)

    from orbitoby import Archive
    from orbitoby.auth import CredentialManager

    creds = CredentialManager(allow_dotenv=True, dotenv_path=PROJECT_ROOT / ".env")
    frames, manifest = [], []
    with Archive(credentials=creds) as archive:
        plan = []
        for s in sats.itertuples():
            for a, b in chunks(s.start, s.end, args.chunk_years):
                have = archive.orbit(norad_id=s.norad_id, start=a, end=b, sync=False)
                plan.append((s.norad_id, a, b, len(have)))
        todo = [x for x in plan if x[3] == 0]
        print(
            f"위성 {len(sats)}기, 구간 {len(plan)}개 "
            f"(이미 일부라도 있는 구간 {len(plan) - len(todo)}개)"
        )
        print(f"새로 요청할 구간 최대 {len(todo)}개 → 약 {len(todo) * args.pause / 60:.0f}분 이상")
        if args.dry_run:
            for nid, a, b, n in plan:
                print(f"  {nid:>6}  {a} → {b}  로컬 {n}개")
            return

        for i, (nid, a, b, _) in enumerate(plan):
            t0 = time.time()
            try:
                df = archive.orbit(norad_id=nid, start=a, end=b, sync=True)
                status = "ok"
            except Exception as exc:  # 위성 하나 실패해도 나머지는 계속
                df, status = pd.DataFrame(), f"{type(exc).__name__}: {exc}"
            print(
                f"[{i + 1}/{len(plan)}] {nid} {a}→{b}: {len(df)}개 "
                f"{'' if status == 'ok' else status}"
            )
            frames.append(df)
            manifest.append(
                {
                    "norad_id": nid,
                    "start": str(a),
                    "end": str(b),
                    "records": len(df),
                    "status": status,
                }
            )
            if time.time() - t0 > 1.0 and i + 1 < len(plan):  # 실제 네트워크 요청이 있었으면 쉰다
                time.sleep(args.pause)

    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = out.drop_duplicates("gp_id").sort_values(["norad_id", "epoch"])
    path = pd.io.common.stringify_path(args.out)
    out.to_parquet(path, index=False)
    summary = out.groupby("norad_id").agg(
        name=("object_name", "last"),
        records=("gp_id", "size"),
        first=("epoch", "min"),
        last=("epoch", "max"),
    )
    with open(f"{path}.manifest.json", "w") as f:
        json.dump(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "orbitoby": version("orbitoby"),
                "source": "Space-Track GP_HISTORY via Orbitoby",
                "requests": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(summary.to_string())
    print(f"\n저장: {path}  ({len(out)}개 레코드)")


if __name__ == "__main__":
    main()
