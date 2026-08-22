"""평상시 드리프트 변동폭 추정.

로드맵이 3단계 잔여 과제로 예고한 "드리프트 지표의 경보 임계 기준에 통계적
근거 부재"를 이 데이터 안에서 닫으려는 시도다.

문제. m6 -> m7 비교는 관측 지점이 두 개뿐이라, PSI 0.15가 나와도 그게 큰지
작은지 판단할 기준이 없다. 통상 쓰이는 0.1 / 0.25 경계는 경험칙이며 이
데이터에서 유도된 값이 아니다.

접근. month 0~5의 인접 월 쌍(m0-m1, m1-m2, ... m4-m5)에서 같은 지표를 계산해
"드리프트가 없다고 볼 수 있는 상태의 변동폭"을 얻는다. m6 -> m7 값이 이 범위를
벗어나면 그때는 근거 있는 경보가 된다.

여기서 유도한 drift_psi_alert_threshold 를 모델 run에 함께 기록한다.
판정 임계값을 코드 상수로 두지 않고 run에 남기는 규약을 드리프트 임계에도
적용한 것이다. 유도 근거(평상시 max, mean+3sd)가 같은 run에 있어야
"이 숫자가 어디서 나왔는지"가 run 하나로 설명된다.

홀드아웃은 건드리지 않는다. month 0~5는 이미 학습에 쓴 구간이다.

한계를 두 가지 명시한다.
  - month 0~5는 in-sample이라 점수가 낙관적이다. 인접 월끼리는 둘 다
    in-sample이라 비교가 공정하지만, 여기서 얻은 변동폭이 out-of-sample
    구간의 변동폭과 같다는 보장은 없다.
  - m5 -> m6 은 in-sample -> out-of-sample 전이라 성격이 다르다. 참고로만
    출력하고 변동폭 추정에서는 제외한다.

빈 경계는 기준 분포(logs/baseline_m6.jsonl)의 분위수로 고정한다. 쌍마다 빈을
다시 잡으면 서로 다른 자로 잰 값이 되어 비교가 성립하지 않는다. 7주차에
계산할 m6 -> m7 PSI도 같은 자를 쓴다.

PSI 함수를 src/ 가 아니라 여기 두는 이유. 모니터링 모듈은 7주차에 실제로
필요해질 때 만든다. 지금 만들면 도입 근거를 설명할 수 없다.

사전 조건:
    docker compose up -d mlflow
    python scripts/build_baseline.py     # 빈 경계의 출처

사용법:
    MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/estimate_normal_drift.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_loader  # noqa: E402
from preprocess import MONTH_COL, TRAIN_MONTHS  # noqa: E402
from train import align_categories, load_splits, split_xy  # noqa: E402

from compare_score_paths import score_path_c  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
BASELINE = ROOT / "logs" / "baseline_m6.jsonl"

# 빈이 비면 ln(0)이 발산한다. 기준 쪽은 분위수로 잡아 비지 않지만 비교 쪽은 빌 수 있다.
EPS = 1e-6

# 평상시 최대 PSI에 곱할 여유. 5개 표본의 max를 그대로 임계로 쓰면 정상 상태에서도
# 넘길 수 있다(6번째 정상 쌍이 그보다 클 확률이 낮지 않다). max×2 와 mean+3sd 가
# 같은 대역에 모이므로 2.0을 쓴다.
#
# 2단 임계(주의/유의)는 두지 않는다. 관례는 0.1 / 0.25 두 단계지만
# 표본 5개로 두 단계를 나눌 근거가 없다.
ALERT_MULTIPLIER = 2.0


def load_baseline_scores(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `python scripts/build_baseline.py` 를 실행하세요."
        )
    return np.array(
        [json.loads(l)["score"] for l in path.read_text(encoding="utf-8").splitlines()]
    )


def bin_edges(baseline: np.ndarray, n_bins: int) -> np.ndarray:
    """기준 분포의 분위수로 경계를 만든다. 양 끝은 무한대로 열어둔다."""
    inner = np.quantile(baseline, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.concatenate([[-np.inf], inner, [np.inf]])


def psi(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    """Population Stability Index. a가 기준, b가 비교 대상."""
    pa = np.clip(np.histogram(a, bins=edges)[0] / len(a), EPS, None)
    pb = np.clip(np.histogram(b, bins=edges)[0] / len(b), EPS, None)
    return float(((pb - pa) * np.log(pb / pa)).sum())


def monthly_scores(loaded, months: list[int]) -> dict[int, np.ndarray]:
    train_df, valid_df = load_splits(DATA_DIR)
    train_df, _ = align_categories(train_df, valid_df)

    out: dict[int, np.ndarray] = {}
    for m in months:
        sub = train_df[train_df[MONTH_COL] == m]
        X, _ = split_xy(sub)
        out[m] = score_path_c(X, loaded)
        print(f"  month {m}  {len(X):>8,}행")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="평상시 드리프트 변동폭 추정")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    print(f"tracking uri : {mlflow.get_tracking_uri()}")

    print("[1/5] load baseline (빈 경계의 출처)")
    base = load_baseline_scores(BASELINE)
    edges = bin_edges(base, args.bins)
    print(f"  {len(base):,}개 점수 / 빈 {args.bins}개")

    print("[2/5] load model")
    loaded = model_loader.load()
    t = loaded.threshold
    print(f"  v{loaded.version}  threshold {t:.6f}")

    print("[3/5] score month 0~5 (in-sample)")
    scores = monthly_scores(loaded, TRAIN_MONTHS)
    scores[6] = base

    print("[4/5] 인접 월 쌍")
    print(f"\n  {'쌍':<12}{'PSI':>10}{'경보율':>10}{'Δ경보율':>11}{'p95':>10}")
    print(f"  {'-' * 51}")

    pairs = list(zip(TRAIN_MONTHS[:-1], TRAIN_MONTHS[1:]))
    normal_psi, normal_dalert = [], []

    for a, b in pairs:
        v = psi(scores[a], scores[b], edges)
        ra, rb = (scores[a] >= t).mean(), (scores[b] >= t).mean()
        normal_psi.append(v)
        normal_dalert.append(abs(rb - ra))
        print(f"  m{a} → m{b}    {v:>10.5f}{rb * 100:>9.3f}%"
              f"{(rb - ra) * 100:>+10.3f}%p{np.percentile(scores[b], 95):>10.5f}")

    # in-sample -> out-of-sample 전이. 성격이 달라 추정에서 제외한다.
    a, b = TRAIN_MONTHS[-1], 6
    v56 = psi(scores[a], scores[b], edges)
    ra, rb = (scores[a] >= t).mean(), (scores[b] >= t).mean()
    print(f"  {'-' * 51}")
    print(f"  m{a} → m{b}    {v56:>10.5f}{rb * 100:>9.3f}%"
          f"{(rb - ra) * 100:>+10.3f}%p{np.percentile(scores[b], 95):>10.5f}"
          "   (참고, 추정 제외)")

    print("\n[5/5] 평상시 변동폭 (in-sample 인접 쌍 5개)")

    psi_max = float(np.max(normal_psi))
    threshold = psi_max * ALERT_MULTIPLIER
    # 교차 확인. 서로 다른 방식이 같은 대역을 가리키는지 본다.
    mean_3sd = float(np.mean(normal_psi) + 3 * np.std(normal_psi, ddof=1))

    summary = {
        "drift_normal_psi_median": float(np.median(normal_psi)),
        "drift_normal_psi_max": psi_max,
        "drift_normal_psi_mean_plus_3sd": mean_3sd,
        "drift_normal_alert_diff_max": float(np.max(normal_dalert)),
        "drift_psi_alert_threshold": threshold,
    }
    for k, v in summary.items():
        print(f"  {k:<32} {v:.6f}")

    print(f"\n  임계값 = 평상시 max × {ALERT_MULTIPLIER} = {threshold:.4f}")
    print(f"  교차 확인  mean+3sd = {mean_3sd:.4f}  (같은 대역이면 채택 근거가 된다)")
    print(f"  관례값 0.1 은 평상시 max 의 {0.1 / psi_max:.1f}배로 이 데이터에서는 둔감하다.")
    print("  이 임계는 조사 착수 트리거이며 재학습 트리거가 아니다.")

    if args.no_mlflow:
        print("  [skip] --no-mlflow")
        return

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(
        model_loader.MODEL_NAME, model_loader.MODEL_ALIAS
    )
    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_metrics(summary)
    print(f"  logged -> runs:/{mv.run_id}")


if __name__ == "__main__":
    main()
