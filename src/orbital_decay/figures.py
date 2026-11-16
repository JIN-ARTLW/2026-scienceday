"""결과 그림 (뷰어 화면과 PNG 내보내기가 같은 함수를 쓴다).

각 함수는 matplotlib Figure를 받아 그 안을 채운다. 뷰어는 Qt 캔버스의 Figure를,
내보내기는 새 Figure를 넘긴다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from orbital_decay.analysis import tle_density
from orbital_decay.constants import R_EQ_KM
from orbital_decay.results import Run

# 한글 글꼴: 맥(AppleGothic), 윈도우(Malgun Gothic), 리눅스(Nanum/Noto) 중 설치된 것만 쓴다
_KOREAN_FONTS = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
matplotlib.rcParams["font.family"] = [f for f in _KOREAN_FONTS if f in _installed] + ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OBS_COLOR = "#1f2937"
MODEL_COLORS = {"physics": "#6b7280", "fixed_k": "#2563eb"}
EXTRA_COLORS = ["#ea580c", "#16a34a", "#9333ea", "#db2777", "#0891b2", "#ca8a04"]
MODEL_LABELS = {"physics": "순수 물리모델", "fixed_k": "고정 보정계수"}


def model_color(name: str, names: list[str]) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    others = [n for n in names if n not in MODEL_COLORS]
    return EXTRA_COLORS[others.index(name) % len(EXTRA_COLORS)] if name in others else "#ea580c"


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, f"ML 보정 ({name})")


def _empty(fig: Figure, message: str) -> None:
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#6b7280")
    ax.set_axis_off()


def model_summary(fig: Figure, run: Run) -> None:
    """모델 비교: 선택 결과가 있으면 위성별 skill(95% CI), 없으면 감쇠율 RMSE."""
    fig.clear()
    selection, _ = run.selection()
    names = run.model_names
    if selection is not None:
        ax = fig.add_subplot(111)
        s = selection.set_index("model").sort_values("mean_skill")
        y = np.arange(len(s))
        colors = [model_color(m, list(s.index)) for m in s.index]
        ax.barh(y, s["mean_skill"], color=colors, alpha=0.85)
        ax.errorbar(
            s["mean_skill"],
            y,
            xerr=[s["mean_skill"] - s["skill_ci_low"], s["skill_ci_high"] - s["mean_skill"]],
            fmt="none",
            ecolor=OBS_COLOR,
            capsize=3,
            lw=1,
        )
        ax.axvline(0, color=OBS_COLOR, lw=1)
        ax.set_yticks(y, [model_label(m) if m in MODEL_LABELS else m for m in s.index])
        ax.set_xlabel("skill = 1 − RMSE_모델 / RMSE_물리   (0보다 크면 물리모델보다 좋음)")
        ax.set_title(
            f"모델 선택: 위성 단위 중첩 교차검증 (위성 {run.dataset['norad_id'].nunique()}기)"
        )
    else:
        loso, test = run.table("summary_loso"), run.table("summary_test")
        tables = [(t, lbl) for t, lbl in ((test, "검증 위성"), (loso, "LOSO")) if t is not None]
        if not tables:
            _empty(fig, "요약 표가 없습니다")
            return
        axes = fig.subplots(1, len(tables), squeeze=False)[0]
        for ax, (t, lbl) in zip(axes, tables, strict=True):
            t = t.set_index("model")
            ax.bar(
                [model_label(m) for m in t.index],
                t["rmse_m_day"],
                color=[model_color(m, names) for m in t.index],
            )
            ax.set_ylabel("감쇠율 RMSE [m/day]")
            ax.set_title(f"{lbl} (n={int(t['n'].iloc[0])})")
            ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()


def satellite(fig: Figure, run: Run, norad_id: int) -> None:
    """위성 하나: 고도, 관측·예측 감쇠율, 태양활동."""
    fig.clear()
    ds = run.dataset
    sat = ds[ds["norad_id"] == norad_id].sort_values("t_mid")
    if sat.empty:
        _empty(fig, f"{norad_id}: 자료 없음")
        return
    pred = run.predictions()
    psat = pred[pred["norad_id"] == norad_id].sort_values("t_mid") if pred is not None else None
    names = run.model_names

    ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1.4, 0.9]})
    calib_end = sat.loc[sat["is_calib"], "t_end"].max()
    for ax in (ax1, ax2, ax3):
        if pd.notna(calib_end):
            ax.axvspan(sat["t_start"].min(), calib_end, color="#e5e7eb", zorder=0)

    ax1.plot(sat["t_mid"], sat["alt_mid_km"], color=OBS_COLOR, lw=1.5)
    ax1.set_ylabel("평균고도 [km]")
    bc = sat["bc_est"].iloc[0]
    ax1.set_title(f"NORAD {norad_id}   BC = {bc:.4f} m²/kg   (회색: BC 추정 구간)")

    ax2.scatter(
        sat["t_mid"], -sat["obs_rate_km_day"] * 1e3, s=10, color=OBS_COLOR, label="관측 (TLE)"
    )
    if psat is not None and not psat.empty:
        for m in names:
            ax2.plot(
                psat["t_mid"],
                -psat[f"pred_{m}"] * 1e3,
                lw=1.4,
                color=model_color(m, names),
                label=model_label(m),
                ls="--" if m == "physics" else "-",
                zorder=3 if m == "physics" else 2,
            )
    ax2.set_ylabel("감쇠율 −da/dt [m/day]")
    ax2.legend(fontsize=8, loc="upper left", ncols=2)

    ax3.plot(sat["t_mid"], sat["f107_obs"], color="#ca8a04", lw=0.8, alpha=0.6, label="F10.7")
    ax3.plot(sat["t_mid"], sat["f107_trail81"], color="#92400e", lw=1.6, label="F10.7 81일 평균")
    ax3.set_ylabel("F10.7 [sfu]")
    ax3b = ax3.twinx()
    ax3b.bar(sat["t_mid"], sat["ap_daily"], width=8, color="#7c3aed", alpha=0.3)
    ax3b.set_ylabel("Ap", color="#7c3aed")
    ax3.legend(fontsize=8, loc="upper left")
    fig.tight_layout()


def satellite_metrics(run: Run, norad_id: int) -> pd.DataFrame | None:
    t = run.table("summary_by_satellite")
    loso = run.table("summary_loso")
    pred = run.predictions()
    if pred is not None:
        from orbital_decay.evaluation import rate_metrics

        p = pred[pred["norad_id"] == norad_id]
        if not p.empty:
            rows = [
                {"model": m, **rate_metrics(p["obs_rate_km_day"], p[f"pred_{m}"])}
                for m in run.model_names
            ]
            return pd.DataFrame(rows).set_index("model")
    if t is not None and loso is None:
        return t[t["norad_id"] == norad_id].set_index("model")
    return None


def propagation(fig: Figure, run: Run, norad_id: int) -> None:
    """보정 구간 끝에서 출발한 모델별 궤도 전파 vs 관측 고도."""
    fig.clear()
    curves = run.table("propagation_curves")
    if curves is None or norad_id not in set(curves["norad_id"]):
        _empty(fig, "이 위성의 전파 결과가 없습니다 (검증 위성만 계산됨)")
        return
    ds = run.dataset
    sat = ds[(ds["norad_id"] == norad_id) & ~ds["is_calib"]].sort_values("t_mid")
    c = curves[curves["norad_id"] == norad_id]
    names = list(dict.fromkeys(c["model"]))
    t_end = c["time"].max()
    ax, axe = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    obs = sat[sat["t_mid"] <= t_end]
    ax.scatter(obs["t_mid"], obs["alt_mid_km"], s=10, color=OBS_COLOR, label="관측 (TLE)", zorder=3)
    for m in names:
        cm = c[c["model"] == m]
        ax.plot(
            cm["time"],
            cm["alt_km"],
            color=model_color(m, names),
            lw=1.6,
            label=model_label(m),
            ls="--" if m == "physics" else "-",
            zorder=3 if m == "physics" else 2,
        )
        err = cm["alt_km"].to_numpy() - np.interp(
            cm["time"].astype("int64"), obs["t_mid"].astype("int64"), obs["alt_mid_km"]
        )
        axe.plot(cm["time"], err, color=model_color(m, names), lw=1.2)
    axe.axhline(0, color=OBS_COLOR, lw=0.8)
    ax.set_ylabel("평균고도 [km]")
    axe.set_ylabel("예측 − 관측 [km]")
    ax.set_title(f"NORAD {norad_id}: 보정 구간 끝에서 출발한 궤도 전파 (관측 TLE는 쓰지 않음)")
    ax.legend(fontsize=8)
    fig.tight_layout()


def solar(fig: Figure, run: Run) -> None:
    """태양활동 관계: F10.7 지연 상관, TLE 역산 밀도 vs F10.7."""
    fig.clear()
    ax1, ax2 = fig.subplots(1, 2)
    lag = run.table("lag_correlation_f107")
    if lag is not None:
        ax1.plot(lag["lag_days"], lag["r"], marker="o", ms=3, color="#92400e")
        best = lag.loc[lag["r"].idxmax()]
        ax1.axvline(best["lag_days"], color="#6b7280", ls="--", lw=1)
        ax1.set_title(
            "TLE 역산 밀도와 F10.7(지연 적용)의 상관\n"
            f"최대 r = {best['r']:.2f} @ {int(best['lag_days'])}일"
        )
        ax1.set_xlabel("지연 [일]")
        ax1.set_ylabel("상관계수 r (장기 추세 제거)")
    ds = run.dataset
    rho = tle_density(ds)
    ok = rho.notna()
    sc = ax2.scatter(
        ds.loc[ok, "f107_trail81"],
        rho[ok],
        c=ds.loc[ok, "alt_mid_km"],
        s=6,
        cmap="viridis_r",
        alpha=0.7,
    )
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
    ax2.set_xlabel("F10.7 81일 후행 평균 [sfu]")
    ax2.set_ylabel("TLE 역산 궤도평균 밀도 [kg/m³]")
    ax2.set_title("태양활동이 강할수록 같은 고도의 밀도가 커진다")
    fig.colorbar(sc, ax=ax2, label="평균고도 [km]")
    fig.tight_layout()


def simulation(fig: Figure, curves: pd.DataFrame, info: dict, title: str) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    names = list(dict.fromkeys(curves["model"]))
    for m in names:
        c = curves[curves["model"] == m]
        life = info["lifetime_days"].get(m)
        lbl = model_label(m) + (
            f" — 수명 {life / 365.25:.1f}년" if life else " — 기간 내 재진입 안 함"
        )
        ax.plot(
            c["time"],
            c["alt_km"],
            color=model_color(m, names),
            lw=1.8,
            label=lbl,
            ls="--" if m == "physics" else "-",
            zorder=3 if m == "physics" else 2,
        )
    end = info.get("uses_analog_after")
    if end is not None and end < curves["time"].max():
        ax.axvspan(end, curves["time"].max(), color="#fef3c7", zorder=0)
        ax.text(
            end,
            ax.get_ylim()[1],
            "  이후: 과거 주기(SC24) 재사용 시나리오",
            va="top",
            fontsize=8,
            color="#92400e",
        )
    ax.set_ylabel("평균고도 [km]")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()


def orbit_shape(fig: Figure, alt_km: float, inc_deg: float) -> None:
    """원궤도 모양(실제 축척)과 하루 지상궤적."""
    from orbital_decay.density import _sample_geometry
    from orbital_decay.orbit import make_satrec

    fig.clear()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    th = np.linspace(0, 2 * np.pi, 400)
    ax1.fill(R_EQ_KM * np.cos(th), R_EQ_KM * np.sin(th), color="#bfdbfe")
    r = R_EQ_KM + alt_km
    ax1.plot(r * np.cos(th), r * np.sin(th), color="#ea580c", lw=1.5)
    ax1.set_aspect("equal")
    ax1.set_title(
        f"궤도 반지름 {r:,.0f} km (지구 {R_EQ_KM:,.0f} km)\n"
        f"고도 {alt_km:.0f} km는 지구 반지름의 {alt_km / R_EQ_KM:.1%}"
    )
    ax1.set_axis_off()

    n = np.sqrt(3.986004418e14 / (r * 1e3) ** 3) * 86400 / (2 * np.pi)
    t0 = pd.Timestamp("2024-01-01")
    sat = make_satrec(
        pd.Series(
            {
                "epoch": t0,
                "mean_motion": n,
                "eccentricity": 0.0005,
                "inclination_deg": inc_deg,
                "raan_deg": 0.0,
                "arg_pericenter_deg": 0.0,
                "mean_anomaly_deg": 0.0,
                "bstar": 0.0,
            }
        )
    )
    times = t0 + pd.to_timedelta(np.linspace(0, 1, 1500), unit="D")
    lon, lat, _ = _sample_geometry(sat, pd.DatetimeIndex(times))
    brk = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    for seg_lon, seg_lat in zip(np.split(lon, brk), np.split(lat, brk), strict=True):
        ax2.plot(seg_lon, seg_lat, color="#ea580c", lw=0.8)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    ax2.set_xlabel("경도 [°]")
    ax2.set_ylabel("위도 [°]")
    ax2.set_title(f"하루 지상궤적 (경사각 {inc_deg:.1f}°, 하루 {n:.1f}바퀴)")
    ax2.grid(alpha=0.3)
    ax2.set_aspect("equal")
    fig.tight_layout()


def export_all(run: Run, out_dir: Path | None = None, dpi: int = 160) -> list[Path]:
    """창 없이 주요 그림을 PNG로 저장한다. 반환: 저장한 파일 목록."""
    out_dir = Path(out_dir) if out_dir else run.path / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(name, draw, *args, size=(10, 6)):
        fig = Figure(figsize=size)
        draw(fig, *args)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi)
        saved.append(path)

    save("model_summary", model_summary, run)
    save("solar_activity", solar, run, size=(12, 5))
    curves = run.table("propagation_curves")
    prop_ids = set(curves["norad_id"]) if curves is not None else set()
    for norad_id in sorted(run.dataset["norad_id"].unique()):
        save(f"satellite_{norad_id}", satellite, run, int(norad_id), size=(11, 8))
        if norad_id in prop_ids:
            save(f"propagation_{norad_id}", propagation, run, int(norad_id), size=(10, 6))
    return saved
