#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Use the existing environment without syncing unrelated project packages.
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import marimo, matplotlib, pandas' >/dev/null 2>&1; then
    exec .venv/bin/python -m marimo edit research/01_gp_history.py "$@"
fi

printf '%s\n' 'Marimo 발표 환경이 없습니다. 프로젝트에서 아래 명령을 실행하세요:' >&2
printf '%s\n' '  uv venv --python 3.12' >&2
printf '%s\n' '  uv pip install --python .venv/bin/python "marimo==0.24.2" "matplotlib>=3.11.2" "pandas>=3.0.6"' >&2
exit 1
