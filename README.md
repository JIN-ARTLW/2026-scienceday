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

## 궤도 감쇠 예측 모델 (`src/orbital_decay`)

순수 물리모델 / 고정 보정계수 모델 / ML 보정모델을 같은 자료로 학습·비교합니다.
입력은 Orbitoby `orbit_elements` 표(열 이름 그대로)입니다.

```bash
# 가상 위성으로 전체 파이프라인 확인 (정답 BC·밀도 편향을 아는 자료)
.venv/bin/python scripts/run_models.py --synthetic --loso

# Orbitoby DuckDB에서 바로
.venv/bin/python scripts/run_models.py --orbitoby-db ~/.orbitoby/warehouse/archive.duckdb --loso

# 내보낸 parquet/csv + 위성 제원(norad_id,mass_kg,area_m2, 선택)
.venv/bin/python scripts/run_models.py --elements data/processed/orbit_elements.parquet \
    --metadata data/sample/satellites.csv
```

결과는 `data/processed/models/<시각>/`에 저장됩니다 (요약 CSV, 창별 예측, 전파 곡선, `run.json`).

### 처리 단계

1. `orbit.prepare_history`: SGP4 초기화로 Brouwer 평균 반장축 계산, 이동중앙값 기준 이상치 제거.
   Orbitoby의 `semimajor_axis_km`(Kozai 평균운동 기반)와 약 1 km 다르므로 섞지 않습니다.
2. `orbit.decay_windows`: 14일 창마다 반장축 직선 적합 → 관측 감쇠율 da/dt.
3. `density`: 실제 궤도 경로를 따라 NRLMSIS 2.1 밀도를 표본 추출한 궤도 평균 밀도 표 (고도층 7개, 캐시).
4. `physics`: da/dt = −BC·ρ·√(μa). 각 위성의 **앞 365일(보정 구간)** 으로만 BC를 최소제곱 추정.
5. `models`
   - `PhysicsModel`: BC × MSIS 물리 예측
   - `FixedCorrectionModel`: 학습 위성에서 구한 배율 k 하나를 곱함
   - `MLCorrectionModel`: 배율을 특징(고도, 경사각, BC, F10.7·Ap 후행 평균 등)의 함수로 학습
     (XGBoost, 불가하면 scikit-learn HistGradientBoosting). 목표 = 관측/물리, 가중치 = 물리²
     → 감쇠율 제곱오차 최소화.
6. `evaluation`: 위성 단위 2/3·1/3 분할, 위성 하나씩 빼는 교차검증(`--loso`),
   태양활동 조건별 오차, 보정 구간 끝에서의 궤도 전파 고도 오차(30/90/180/365일).
7. `analysis.lag_correlation`: TLE 역산 밀도와 F10.7의 지연 상관.

### 주의

- ML 특징에는 미래 정보가 없는 후행 평균만 씁니다. MSIS 입력의 81일 중심 평균 F10.7은 물리모델에만 쓰입니다.
- 평균고도 250 km 아래(재진입 직전) 창은 기본으로 제외합니다 (`--min-alt-km`).
- XGBoost를 쓰려면 macOS에서 `brew install libomp`가 필요합니다. 없으면 자동으로 scikit-learn을 씁니다.

### ML 모델 선택 (`scripts/select_model.py`)

후보(선형, 선형+상호작용, 가중 kNN, SVR, 가우시안 과정, 랜덤포레스트, Extra Trees,
HistGradientBoosting, MLP, 그리고 설치돼 있으면 XGBoost·LightGBM·CatBoost)를 같은 조건에서 비교합니다.

```bash
.venv/bin/python scripts/select_model.py data/processed/models/<run>/dataset.parquet
```

- 바깥 루프: 위성 하나씩 빼기(LOSO), 안쪽 루프: 학습 위성만으로 GroupKFold 하이퍼파라미터 선택
- 위성별 skill = 1 − RMSE_모델 / RMSE_순수물리, 부트스트랩 95% CI
- 선택 규칙: 최고 후보와 위성별 RMSE가 유의하게 다르지 않은(Wilcoxon p ≥ 0.05) 후보 중 가장 단순한 것.
  순수 물리모델도 후보라서, 보정이 일반화된다는 근거가 없으면 물리모델이 선택됩니다.
- 결과: `selection/report.md` (방법, 표, 선택 근거, 후보에서 뺀 방법과 이유)

## 맥·윈도우 공동 작업 (Git + Google Drive)

- **코드**: Git(GitHub)으로만 공유. 저장소·`.venv`는 Google Drive 밖(각자 로컬)에 둔다.
- **데이터·결과**: 각 컴퓨터의 `.env`에서 `SCIENCEDAY_DATA_DIR`를 같은 Drive 폴더로 지정한다
  (`.env.example` 참고). 모델 결과는 `<Drive>/…/data/processed/models/<시각>_<컴퓨터>/`에 쌓인다.
- Orbitoby DuckDB는 Drive 안에서 직접 쓰지 않는다. 로컬에서 수집한 뒤 스냅샷을 복사해 읽기 전용으로 쓴다.
- 계산할 컴퓨터에서는 Drive 폴더를 "오프라인 사용 가능"으로 설정한다 (스트리밍 모드는 첫 읽기가 느림).

## 결과 뷰어 (`scripts/viewer.py`)

결과 폴더(`<데이터 폴더>/processed/models/`)를 읽어 창으로 보여줍니다. 계산은 윈도우 노트북에서 하고,
Google Drive로 동기화된 결과를 맥에서 열어도 됩니다.

```bash
uv sync --extra viewer --inexact            # 처음 한 번 (PySide6 설치, 다른 패키지는 건드리지 않음)
uv run python scripts/viewer.py             # 창 열기 (최신 결과)
uv run python scripts/viewer.py --export    # 창 없이 그림 PNG 저장 → <결과 폴더>/figures/
```

| 탭 | 내용 |
|---|---|
| 요약 | 실행 정보, 모델 선택 근거(`selection/report.md`), 모델별 skill·95% CI |
| 위성별 | 평균고도, 관측·모델별 감쇠율, F10.7·Ap (회색 = BC 추정 구간) |
| 궤도 전파 | 보정 구간 끝에서 출발한 모델별 고도 예측 vs 관측, 기간별 오차 |
| 태양활동 | TLE 역산 밀도와 F10.7 지연 상관, 밀도-F10.7 산점도 |
| 시뮬레이션 | 고도·경사각·질량·단면적·시작일·태양활동 배율 → 모델별 감쇠 곡선과 수명, 궤도 모양·지상궤적 |
| 시뮬레이션 › 3D 궤도 | GMAT 스타일 3D: 재생·속도·시간 슬라이더, ECI/ECEF 전환, 모델별 고도 곡선, 고도 과장, 궤적 꼬리·지상궤적 (마우스로 회전·확대) |

- 시뮬레이션은 선택한 결과 폴더의 학습 모델(`models.joblib`)을 씁니다.
  윈도우에서 XGBoost로 학습한 결과를 libomp 없는 맥에서 열면 ML 곡선만 빠지고 나머지는 동작합니다.
- 3D 지구 텍스처 `assets/earth.jpg`: NASA Visible Earth *Blue Marble* (land_shallow_topo_2048,
  NASA Goddard Space Flight Center, Reto Stöckli; 퍼블릭 도메인).
  위치 정확도 검증(`tests/test_orbit3d.py`):
  - 뷰어의 좌표 변환 → skyfield 직하점과 경도 < 0.01°, 위도 < 0.03° (측지위도 보정 포함)
  - 텍스처 정합 → 작은 섬·좁은 바다 포함 기준점 24곳 육지/바다 판별 (±1° 밀면 틀리기 시작)
  - 지구 메시 0.5° 간격 → 해안선 위치 오차가 텍스처 픽셀(0.18°) 수준
  - 시뮬레이션 궤도의 교점 경도(RAAN)는 가정값(0°)이라, 특정 실제 위성의 위치가 아니라
    '이런 궤도라면 이렇게 지나간다'는 계산입니다.
- 태양·지자기 자료가 끝난 뒤(현재 2026-11-12 이후)는 과거 주기(SC24)를 다시 쓰는 시나리오이며,
  그래프에 노란 영역으로 표시됩니다.
