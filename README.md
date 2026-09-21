# 2026 사이언스데이

## Marimo 발표용 선행연구 리뷰

프로젝트 폴더에서 실행합니다.

```bash
./scripts/marimo.sh
```

`research/01_gp_history.py`를 Marimo 편집기로 엽니다. 코드 셀은 기본적으로 접혀 있으며,
슬라이더를 바꾸면 연결된 계산과 그래프가 갱신됩니다. 경로에 공백이 있거나 다른 폴더에서
스크립트를 호출해도 프로젝트 기준으로 실행합니다. 추가 CLI 옵션도 전달할 수 있습니다.

```bash
./scripts/marimo.sh --headless --port 2718
```

발표용 실행 화면만 열려면:

```bash
.venv/bin/python -m marimo run research/01_gp_history.py
```

### 최소 실행 환경

Python 3.12와 직접 사용하는 패키지 **marimo, matplotlib, pandas**만 필요합니다.
기존 `.venv` 및 `pyproject.toml`/`uv.lock`의 설치 환경을 우선 사용하며, 실행 스크립트는
패키지 동기화나 업데이트를 하지 않습니다. 새 환경을 만들 때만:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python "marimo==0.24.2" "matplotlib>=3.11.2" "pandas>=3.0.6"
```

전체 연구 환경 복원은 기존 `uv sync --locked`를 사용할 수 있지만, 발표 노트북에는
나머지 연구 패키지나 Streamlit, Space-Track 인증, Windows 서버가 필요하지 않습니다.
첫 실행 시 로컬 폰트 캐시 생성으로 잠시 시간이 걸릴 수 있습니다.

### 화면과 출처

Ashruf et al. (2026), *Characterizing solar cycle influence on long-term orbital
 deterioration of low-earth orbiting space debris*, Front. Astron. Space Sci. 13:1797886.
[논문 DOI](https://doi.org/10.3389/fspas.2026.1797886), CC BY.
업로드된 `2026_Ashruf_SolarCycle_LEO_OrbitalDecay.pdf`의 본문과 표를 직접 대조했습니다.

1. 태양활동 → 열권 밀도 → 항력 → 궤도감쇠: pp. 2–6, Figures 1–2.
2. SC22/23/24 평균 peak decay: Table 2(p. 4)의 17개 값을 전사해 평균 계산.
   Figure 4(p. 8)의 본문 요약값과 표의 반올림 수치에서 재계산한 값을 구분합니다.
3. R² 비교: Table 4(p. 8)의 평균과 물체 간 표준편차. 전체 Kp 표도 포함합니다.
4. BC = Cd A/m 및 민감도 슬라이더: p. 3의 정의와 Eq. 3(p. 4).
   밀도·속도·시간은 임의 고정값이며, 속도 변화 크기만 시연합니다.
5. 보정 후 예측 일치와 고경사 물체의 한계: pp. 9–11, Table 5, Figure 6.
   우리 연구 질문과 향후 검증 계획으로 연결합니다.

논문 기반 결과(Source-derived), 가정 시연(Illustrative), 연구 계획(Planned)을
화면에 구분했습니다. 이 노트북은 원자료 재분석·MSIS 실행·궤도 전파를 수행하지 않습니다.
실행 시 논문 파일이나 네트워크 자료 다운로드가 필요하지 않습니다.

### 검증 명령

```bash
.venv/bin/python -m marimo check research/01_gp_history.py
.venv/bin/python research/01_gp_history.py
```

[Marimo 공식 문서](https://docs.marimo.io/)에서 편집·실행 모드를 확인할 수 있습니다.
