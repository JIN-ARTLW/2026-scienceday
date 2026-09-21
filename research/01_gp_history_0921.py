import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    return mo, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    26/09/21 진예서
    ~귀여운 마리모와 함께하는 발표~
    ### Topic: 사이언스 데이 주제 선정 및 주제의 과학적 배경에 관하여...
    ---

    ## **연구 주제:** 태양 활동에 따른  저궤도 위성 궤도 감쇠율 분석
    고도 및 위성 특성에 따른 영향의 차이와 궤도 감쇠 예측 모델 연구
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **연구 동기**
     저궤도(LEO)를 비행하는 인공위성은 완전한 진공 상태에 있지는 않다. 특히 300~800km영역(HST, Starlink 등)은 상층대기, 특히 열권과 그 상부 영역의 희박한 중성대기가 위성의 운동에 영향을 줄 수 있는 구간이다. 위성은 이러한 대기와의 충돌로 지속적인 항력을 받아 에너지를 잃고 고도가 감소하는 궤도 감쇠를 겪는다.
     상층대기 밀도는 태양활동에 따라 일정치 않게 변화한다. 태양의 복사가 증가하면 열권이 가열 및 팽창하여 위성 고도에서의 대기밀도가 증가하고, 지자기폭풍 역시 추가적인 에너지를 상층대기에 공급한다. 실게 관측에서도(GRACE-FO) 태양 주기 25상승기의 복사가 열권 밀도 증가의 주요 요인으로 나타났으며, CHAMP 연구에서도 지자기 폭풍이 강할수록 궤도감쇠가 증가하고 낮은 고도에서 그 영향이 더 크게 나타났다.
     따라서 본 연구에서는 태양활동, 상층대기 밀도 변화, 대기항력 변화, 저궤도 위성의 궤도 감쇠라는 상관관계를 실제 자료를 통해 분석하고. 고도와 위성의 물리적 특성에 따라 그 영향이 어떻게 달라지는지 연구하고자 한다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **태양활동과 저궤도 위성의 궤도 감쇠**

    선행연구에서 제시되는 기본적인 물리적 연결은 다음과 같다.

    \[
    \text{Solar Activity} \uparrow
    \]

    \[
    \Downarrow
    \]

    \[
    \text{EUV} \uparrow
    \rightarrow
    \text{Thermospheric Heating / Expansion}
    \]

    \[
    \Downarrow
    \]

    \[
    \rho_{\mathrm{thermosphere}} \uparrow
    \rightarrow
    \text{Atmospheric Drag} \uparrow
    \rightarrow
    \text{Orbital Decay} \uparrow
    \]

    즉, 태양활동이 위성을 직접 아래로 끌어당기는 것이 아니라,
    태양 복사가 상층대기를 가열, 팽창시키고 대기밀도를 증가시켜
    위성이 받는 항력을 증가시키는 과정이 핵심이다.

    +
    **TLE**: 위성의 궤도 정보 데이터
    **F10.7**: 태양이 방출하는 파장 10.7 cm, 주파수 약 2.8 GHz의 전파 세기로 EUV/자외선 계열의 복사활동과 함께 움직이는 지표.(대리지표로써 사용)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **연구 목적**

    본 연구의 목적은 태양활동과 저궤도 인공위성의 궤도 감쇠 사이의 관계를 물리적으로 분석하고 실제 위성 자료를 통해 검증하는 것이다.
     이를 위해 태양활동을 나타내는 지표와 상층대기의 상태를 조사하고, 실제 저궤도 위성의 시간에 따른 궤도 변화를 비교한다. 이를 통해 태양 활동이 강한 시기에 위성의 궤도 감쇠율이 어떻게 달라지는지 확인한다.
     또한 위성의 초기 고도와 질량, 단면적 등의 차이에 따라 동일한 태양활동 변화에 대한 궤도 감쇠 반응이 어떻게 달라지는지를 분석한다.
     최종적으로 이러한 관계를 바탕으로 태양활동과 위성의 물리적 조건을 입력하면 예상되는 궤도 감쇠를 계산할 수 있는 예측모델을 구축하고 실제 위성자료를 통해 모델의 예측 가능성을 검증하고자 한다.
     주요 연구 질문은 다음과 같다.
    -	태양 활동이 강해질수록 열권 밀도와 위성의 궤도 감쇠율은 어떻게 변화하는가?
    -	동일한 태양활동의 변화에서도 위성의 고도에 따라 영향의 크기가 달라지는가?
    -	질량과 유효 단면적 등 위성의 물리적 특성에 따라 어떻게 달라지는가?
    -	이러한 관계를 이용하여 다양한 저궤도 위성의 감쇠를 어디까지 예측할 수 있는가?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="text-align: center;">
          <img
            src="https://www.frontiersin.org/files/Articles/1797886/xml-images/fspas-13-1797886-g001.webp"
            style="width: 50%;"
          />
        </div>

    (참고문헌 피겨1에서 발췌)
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    decay_df = pd.DataFrame({
        "SATCAT": [
            22, 29, 45, 46, 115, 162, 227, 228, 262,
            309, 397, 716, 720, 733, 734, 876, 877
        ],
        "SC22": [
            -1.42, -0.52, -0.39, -0.73, -0.90,
            -0.24, -0.42, -1.04, -1.15,
            -0.53, -0.66, -0.47, -0.52,
            -0.25, -0.17, -0.36, -0.32
        ],
        "SC23": [
            -1.40, -0.46, -0.30, -0.61, -0.93,
            -0.18, -0.33, -1.20, -1.10,
            -0.45, -0.63, -0.40, -0.43,
            -0.18, -0.12, -0.27, -0.25
        ],
        "SC24": [
            -0.81, -0.17, -0.11, -0.23, -0.44,
            -0.06, -0.11, -0.71, -0.63,
            -0.16, -0.27, -0.15, -0.15,
            -0.07, -0.04, -0.09, -0.09
        ]
    })

    cycle_summary = (
        decay_df[["SC22", "SC23", "SC24"]] #태양주기 선택
        .agg(["mean", "median"]) #평균 및 중앙값 선택
        .T
        .rename(
            columns={
                "mean": "Mean decay rate [m/h]",
                "median": "Median decay rate [m/h]",
            }
        )
    )

    cycle_summary
    return (cycle_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **단위** : \(m/h\)
    각 주기 별 감쇠정도 평균/중앙값

    **논문에서**:
    - SC22 평균 −0.59 m/h
    - SC23 −0.54 m/h
    - SC24 −0.25 m/h
    """)
    return


@app.cell
def _(cycle_summary, plt):
    fig_cycle, ax_cycle = plt.subplots(figsize=(7, 4))

    ax_cycle.bar(
        cycle_summary.index,
        cycle_summary["Mean decay rate [m/h]"],
    )

    ax_cycle.axhline(0, linewidth=1)

    ax_cycle.set_title(
        "Mean peak orbital decay rate across solar cycles"
    )
    ax_cycle.set_xlabel("Solar cycle")
    ax_cycle.set_ylabel("Mean decay rate [m/h]")

    fig_cycle
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **x축**: 태양 활동 주기
    **y축**: 감쇠율

    SC22와 SC23에서 SC24보다 큰 평균 peak 감쇠가 나타났으며, 이 선행연구에서는 **상대적으로 강한 태양활동 주기에서 더 큰 궤도 감쇠가 나타나는 경향**을 확인할 수 있다.
    """)
    return


@app.cell
def _(pd, plt):
    r2_df = pd.DataFrame({
        "Index": [
            "F10.7",
            "SSN",
            "Dst",
            "Ap",
            "AE",
        ],
        "Mean R² [%]": [
            74.9,
            66.9,
            22.2,
            18.3,
            1.2,
        ],
    })

    fig_r2, ax_r2 = plt.subplots(figsize=(7, 4))

    ax_r2.bar(
        r2_df["Index"],
        r2_df["Mean R² [%]"],
    )

    ax_r2.set_ylim(0, 100)

    ax_r2.set_title(
        "Orbital decay vs solar/geomagnetic"
    )
    ax_r2.set_xlabel("Activity index")
    ax_r2.set_ylabel("Mean R² [%]")

    fig_r2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 궤도 감쇠율과 태양/지자기 활동 지표의 관계:
    평균 결정계수 \(R^2\)이 0에 가까울수록 관계가 약하고, 1에 가까울수록 관계가 강함.(그래프에서는 백분율로 표시)
    여기서는
    - F10.7: 74.9% (태양 10.7 cm 전파 플럭스)
    - SSN: 66.9% (흑점수)
    - Dst: 22.2% (지자기폭풍 강도)
    - Ap: 18.3% (전 지구적 지자기 활동 수준)
    - AE: 1.2% (오로라대 전류 활동)

    으로 나타났다. 즉, 장기적인 궤도 감쇠 변화가 지자기 지표들보다 F10.7이나 SSN 같은 태양활동 지표와 훨씬 강한 관계를 보였다.
    """)
    return


@app.cell
def _(mo):
    bc_slider = mo.ui.slider(
        start=0.001,
        stop=0.050,
        step=0.001,
        value=0.010,
        label="질량당 항력 Cd·A/m [m²/kg]",
    )

    bc_slider
    return (bc_slider,)


@app.cell(hide_code=True)
def _(bc_slider, mo):
    rho_demo = 1.0e-13
    velocity_demo = 7500.0
    dt_demo = 3600.0

    delta_v_demo = (
        0.5
        * rho_demo
        * velocity_demo**2
        * bc_slider.value
        * dt_demo
    )

    mo.md(
        rf"""
    ## Bc(탄도계수) 예시

    현재 설정

    - \(\rho = {rho_demo:.1e}\;kg/m^3\)
    - \(v = {velocity_demo:.0f}\;m/s\)
    - \(\Delta t = {dt_demo:.0f}\;s\)
    - \(C_DA/m = {bc_slider.value:.3f}\;m^2/kg\)

    계산된 항력에 의한 속도 변화의 크기:

    \[
    |\Delta v|
    =
    {delta_v_demo:.6f}\;m/s
    \]

    \(C_DA/m\) 값을 증가시키면
    같은 대기환경에서도 항력의 효과가 증가한다.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    즉 같은 태양활동 하에서도 물리량에 따라 물체의 감쇠율이 달라질 수 있다!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **연구 방법**
     본 연구에서는 실제 태양활동 자료와 저궤도 물체의 궤도자료를 수집하여 동일한 시간축에서 비교/분석한다.
     위성의 TLE에서 시간에 따른 궤도 고도 및 감쇠율을 계산하고, 같은 기간의 F10.7과 상층대기 밀도 변화와 비교한다. 이후 위성들을 물리량에 따라 구분하여 태양활동 변화에 대한 감쇠 민감도를 분석한다. 2026년 연구에서도 TLE와 F10.7, 질량/단면적 자료 등을 결합하여 실제 궤도감쇠를 모델링하고 관측자료와 비교하였다.
     모델 검증은 모델 작성에 사용하지 않은 별도의 위성 또는 시간구간을 이용한다. 실제 TLE에서 관측된 감쇠율과 모델의 예측값을 비교하여 RMSE등의 오차를 구하고. 가능한 경우 위성에서 직접 얻어진 상층대기 밀도자료와도 비교한다. 최근 연구에서도 독립된 관측자료를 이용한 검증이 열권 밀도 및 항력 예측의 핵심 과정으로 사용되고 있다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **연구 내용**
     본 연구에서는 먼저 태양활동 변화와 상층대기 밀도 및 저궤도 위성의 감쇠율 사이의 관계를 분석한다. 특히 태양활동이 높은 시기에 대기밀도와 궤도 감쇠가 함께 증가하는지 실제 자료를 통해 확인한다.
     다음으로 서로 다른 고도의 위성을 비교하여 고도에 따른 태양활동 민감도를 분석하고, 질량과 단면적을 이용한 탄도계수를 통해 위성의 물리적 특성에 따른 감쇠 차이를 비교한다.
     최종적으로 이러한 결과를 토대로 태양활동, 고도, 질량 및 단면적 등의 조건을 입력하면 예상되는 상층대기 밀도와 궤도 감쇠율을 계산할 수 있는 모델을 구현한다. 이 모델을 실제 위성 자료와 비교하여 정확도와 한계를 평가/검증 하고, 최종적으로 시뮬레이션 소프트웨어 형태로 구현한다.

    | 변수 | 의미 | 역할 |
    |---|---|---|
    | F10.7 | 태양활동 지표 | 태양 입력 |
    | \(\rho\) | 상층대기 밀도 | 태양활동-항력 연결 |
    | \(h\) | 궤도 고도 | 고도별 민감도 |
    | \(m\) | 위성 질량 | 물체 특성 |
    | \(A\) | 유효 단면적 | 물체 특성 |
    | \(C_D\) | 항력계수 | 항력 반응 |
    | \(dh/dt\) | 감쇠율 | 최종적으로 관측할 결과 |

    결론적으로 본 연구는
    \[
    \boxed{
    F10.7
    \rightarrow
    \rho
    \rightarrow
    \text{대기항력}
    \rightarrow
    \frac{dh}{dt}
    }
    \]로 표현할 수 있으며,
    그리고 이 관계가
    \[
    h,\;m,\;A,\;C_D
    \]에 따라 얼마나 달라지는지를 탐구하는 연구이다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **참고 문헌**
    Ashruf, A. M., Bhaskar, A., Vineeth, C., & Pant, T. K. (2026). Characterizing solar cycle influence on long-term orbital deterioration of low-earth orbiting space debris. Frontiers in Astronomy and Space Sciences, 13.
    ( TLE, F10.7, 탄도계수, MSIS 2.0을 이용해 태양주기와 실제 궤도감쇠를 분석 및 예측. 코드의 데이터도 모두 이 논문 기반.)
    """)
    return


if __name__ == "__main__":
    app.run()
