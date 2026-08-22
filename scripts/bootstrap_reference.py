"""2차 판정 기준선 — 부트스트랩 신뢰구간.

7주차 2차 신호(라벨 도착 후)의 판정 기준을 만든다.

  AUPRC        month 7 값이 이 구간 아래면 "저하"
  구간별 사기율  month 7 값이 이 구간을 벗어나면 "보정 깨짐"

두 신호가 **모두** 벗어날 때만 관계 변화로 결론내고 재학습으로 간다.
하나만 벗어나면 판단 보류다. 재학습은 원인이 파이프라인 버그일 때
상태를 악화시키므로, 오판 비용이 조사 비용보다 훨씬 크다.

왜 month 0~5 변동폭이 아니라 부트스트랩인가.
  AUPRC는 라벨을 직접 쓰는 지표라 in-sample 편향을 정면으로 맞는다.
  train_auprc 0.2507 vs auprc 0.1541 로 62.7% 차이가 나므로,
  학습 구간에서 잰 변동폭은 검증 구간을 대표하지 못한다.
  부트스트랩은 month 6 만으로 "같은 분포에서 다시 뽑으면 얼마나 흔들리나"를
  재므로 이 편향이 없다.

리샘플 크기를 month 7 크기에 맞추는 이유. 표본이 작으면 변동성이 크다.
108,168개씩 뽑아 만든 구간을 96,843행짜리 관측에 적용하면 구간이 실제보다
좁아져 저하 판정이 과하게 나온다.

한계. 부트스트랩은 **표본 변동성**만 잰다. 월 간 자연 변동(계절성 등)은
포함하지 않으므로 구간이 실제보다 좁게 나오고, 판정이 민감한 쪽으로 기운다.
위의 "두 신호 일치" 규칙이 그 안전장치다.

구간 경계는 PSI와 다르다. PSI는 분포 이동을 보므로 균등 10분위가 맞고,
여기는 보정을 보므로 구간마다 사기 건수가 충분해야 한다. 10분위로 나누면
하위 5구간의 사기가 3~22건이라 Poisson 노이즈만 ±23~61% 로 판정이 불가능하다.

사전 조건:
    python scripts/build_baseline.py     # 점수와 임계값의 출처
    docker compose up -d mlflow          # 기록할 때만 필요

사용법:
    MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/bootstrap_reference.py
    ... --n-boot 1000 --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_loader  # noqa: E402
from preprocess import EXPECTED_RAW_ROWS, EXPECTED_SPLIT, TARGET  # noqa: E402

BASELINE = ROOT / "logs" / "baseline_m6.jsonl"
VALID = ROOT / "data" / "processed" / "valid.parquet"

# 홀드아웃 행 수를 새로 적지 않고 기존 상수에서 유도한다.
# 1,000,000 - 794,989 - 108,168 = 96,843
DEFAULT_RESAMPLE = EXPECTED_RAW_ROWS - sum(v["rows"] for v in EXPECTED_SPLIT.values())

# 구간 경계는 (분위수 3개 + 판정 임계값)으로 만든다. 마지막 구간이 운영상
# ALERT 구간과 정확히 일치하므로 "심사 대상자의 사기율이 유지되는가"를 직접 묻게 된다.
INNER_QUANTILES = (0.50, 0.80, 0.90)
BIN_LABELS = ("p0-50", "p50-80", "p80-90", "p90-임계", "임계-100 (ALERT)")

CI = (2.5, 97.5)


def load_baseline() -> tuple[np.ndarray, np.ndarray, float]:
    if not BASELINE.exists():
        raise FileNotFoundError(
            f"{BASELINE} 가 없습니다. 먼저 `python scripts/build_baseline.py` 를 실행하세요."
        )
    recs = [json.loads(l) for l in BASELINE.read_text(encoding="utf-8").splitlines()]
    scores = np.array([r["score"] for r in recs])
    threshold = float(recs[0]["threshold"])

    y = pd.read_parquet(VALID, engine="pyarrow")[TARGET].to_numpy().astype(float)
    if len(y) != len(scores):
        raise ValueError(f"행 수 불일치: 점수 {len(scores):,} vs 라벨 {len(y):,}")

    return scores, y, threshold


def bootstrap(
    scores: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    n_bins: int,
    size: int,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(scores)

    auprc = np.empty(n_boot)
    rates = np.empty((n_boot, n_bins))

    for i in range(n_boot):
        idx = rng.integers(0, n, size=size)
        ys, ss, bs = y[idx], scores[idx], bins[idx]

        auprc[i] = average_precision_score(ys, ss)

        cnt = np.bincount(bs, minlength=n_bins)
        pos = np.bincount(bs, weights=ys, minlength=n_bins)
        rates[i] = np.where(cnt > 0, pos / np.maximum(cnt, 1), np.nan)

        if (i + 1) % 200 == 0:
            print(f"    {i + 1:,}/{n_boot:,}")

    return auprc, rates


def main() -> None:
    parser = argparse.ArgumentParser(description="2차 판정 기준선 부트스트랩")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--resample-size", type=int, default=DEFAULT_RESAMPLE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    print(f"tracking uri : {mlflow.get_tracking_uri()}")

    print("[1/4] load baseline")
    scores, y, threshold = load_baseline()
    print(f"  {len(scores):,}행 / 사기 {int(y.sum()):,}건 / 임계 {threshold:.6f}")

    inner = [float(np.quantile(scores, q)) for q in INNER_QUANTILES] + [threshold]
    edges = np.array([-np.inf, *inner, np.inf])
    bins = np.digitize(scores, edges[1:-1])
    n_bins = len(BIN_LABELS)

    print("[2/4] 구간 (기준선)")
    point_rate = []
    for i, label in enumerate(BIN_LABELS):
        m = bins == i
        r = float(y[m].mean())
        point_rate.append(r)
        print(f"  {label:<20}{int(m.sum()):>9,}행  사기 {int(y[m].sum()):>5,}  {r * 100:>7.3f}%")

    point_auprc = float(average_precision_score(y, scores))
    print(f"  {'AUPRC (전체)':<20}{point_auprc:>28.6f}")

    print(f"[3/4] bootstrap  n={args.n_boot:,}  리샘플 크기={args.resample_size:,}")
    boot_auprc, boot_rates = bootstrap(
        scores, y, bins, n_bins, args.resample_size, args.n_boot, args.seed
    )

    print("[4/4] 신뢰구간 (95%)")
    a_lo, a_hi = np.percentile(boot_auprc, CI)
    print(f"\n  AUPRC   기준선 {point_auprc:.6f}   95% [{a_lo:.6f}, {a_hi:.6f}]")
    print(f"          → month 7 이 {a_lo:.6f} 아래면 '저하'\n")

    print(f"  {'구간':<20}{'기준선':>10}{'하한':>10}{'상한':>10}{'폭':>9}")
    print("  " + "-" * 59)
    bin_ci = []
    for i, label in enumerate(BIN_LABELS):
        lo, hi = np.percentile(boot_rates[:, i], CI)
        bin_ci.append((float(lo), float(hi)))
        width = (hi - lo) / point_rate[i] * 100 if point_rate[i] else float("nan")
        print(f"  {label:<20}{point_rate[i] * 100:>9.3f}%{lo * 100:>9.3f}%"
              f"{hi * 100:>9.3f}%{width:>8.0f}%")

    print("\n  두 신호가 모두 구간을 벗어날 때만 관계 변화로 결론낸다.")

    if args.no_mlflow:
        print("  [skip] --no-mlflow")
        return

    payload = {
        "n_boot": args.n_boot,
        "resample_size": args.resample_size,
        "seed": args.seed,
        "threshold": threshold,
        # 양 끝은 -inf / +inf 이며 JSON이 표현하지 못하므로 안쪽 경계만 남긴다.
        "bin_inner_edges": inner,
        "bin_labels": list(BIN_LABELS),
        "auprc": {"point": point_auprc, "lo": float(a_lo), "hi": float(a_hi)},
        "fraud_rate": [
            {"label": l, "point": p, "lo": lo, "hi": hi}
            for l, p, (lo, hi) in zip(BIN_LABELS, point_rate, bin_ci)
        ],
    }

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(
        model_loader.MODEL_NAME, model_loader.MODEL_ALIAS
    )
    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_dict(payload, "drift_reference.json")
        mlflow.log_metrics(
            {
                "boot_auprc_lo": float(a_lo),
                "boot_auprc_hi": float(a_hi),
            }
        )
    print(f"  logged -> runs:/{mv.run_id}  (drift_reference.json)")


if __name__ == "__main__":
    main()
