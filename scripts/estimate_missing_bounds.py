"""컬럼별 결측률 정상 범위 추정.

docs/drift-response.md 한계 5번을 닫는다.

  > 입력 검증의 결측률·점수 모양 항목에 정량 기준이 없다.

방법은 PSI 임계를 유도할 때와 같다. month 0~6 인접 쌍에서 컬럼별 결측률
변동폭을 재고, 거기서 임계를 유도한다.

PSI 와 다른 점 두 가지.

  1. month 6 을 포함한다. 결측률은 라벨과 무관한 순수 피처 분포라
     in-sample / out-of-sample 구분이 의미가 없다. "몇 %가 -1인가"는
     그 구간을 학습에 썼는지와 무관하다. 쌍이 5개에서 6개로 늘어난다.

  2. 절대 하한을 둔다. 평상시 최대가 0.05%p 인 컬럼은 x2 해도 0.1%p 라,
     그 정도로 경보를 울리면 정보가 없다. 파이프라인 버그가 만드는 결측률
     변화는 계단형(전부 아니면 전무)이지 1%p 미만의 미세 변동이 아니다.
     96,843행 기준 1%p 는 968행이며, 그만큼만 깨지는 파손은 드물다.

대상은 모델 피처 전부다. 결측이 원래 0%인 컬럼도 포함한다 — 파이프라인
버그는 **없던 결측을 만들어내는** 형태로 나타나기 때문이다.

사용법:
    MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/estimate_missing_bounds.py
    ... --floor 1.0 --multiplier 2.0 --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_loader  # noqa: E402
from preprocess import MONTH_COL, TRAIN_MONTHS, VALID_MONTHS  # noqa: E402

from _preflight import require_mlflow  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
OBSERVED_MONTHS = TRAIN_MONTHS + VALID_MONTHS  # 0~6


def missing_by_month(features: list[str]) -> pd.DataFrame:
    """month 0~6 각 구간의 컬럼별 결측률(%)."""
    train = pd.read_parquet(DATA_DIR / "train.parquet", engine="pyarrow")
    valid = pd.read_parquet(DATA_DIR / "valid.parquet", engine="pyarrow")
    df = pd.concat([train, valid], ignore_index=True)

    rows = {}
    for m in OBSERVED_MONTHS:
        sub = df[df[MONTH_COL] == m]
        rows[m] = sub[features].isna().mean() * 100
        print(f"  month {m}  {len(sub):>8,}행")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="컬럼별 결측률 정상 범위 추정")
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--floor", type=float, default=1.0, help="절대 하한 (%p)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    print(f"tracking uri : {mlflow.get_tracking_uri()}")
    require_mlflow()

    print("[1/3] load model (피처 목록의 출처)")
    loaded = model_loader.load()
    features = list(loaded.feature_names)
    print(f"  피처 {len(features)}개")

    print("[2/3] 월별 결측률")
    rate = missing_by_month(features)

    print("[3/3] 인접 쌍 변동폭 → 임계")
    pairs = list(zip(OBSERVED_MONTHS[:-1], OBSERVED_MONTHS[1:]))
    delta = pd.DataFrame(
        {f"m{a}→m{b}": (rate[b] - rate[a]).abs() for a, b in pairs}
    )
    normal_max = delta.max(axis=1)
    # PSI 때와 같은 교차 확인. 두 값이 벌어지면 변동 분포가 한 점에 지배된다는 뜻이다.
    mean_3sd = delta.mean(axis=1) + 3 * delta.std(axis=1, ddof=1)
    bound = np.maximum(normal_max * args.multiplier, args.floor)

    out = pd.DataFrame(
        {
            "m6_결측률": rate[VALID_MONTHS[0]],
            "평상시최대": normal_max,
            "mean_3sd": mean_3sd,
            "임계": bound,
            "하한적용": bound <= args.floor + 1e-12,
        }
    ).sort_values("평상시최대", ascending=False)

    print(f"\n  배수 x{args.multiplier}  절대 하한 {args.floor}%p  (쌍 {len(pairs)}개)")
    print("  max x2 와 mean+3sd 가 벌어지면 한 번의 점프가 임계를 지배한다는 뜻이다.\n")
    print(f"  {'컬럼':<32}{'m6':>9}{'평상시최대':>11}{'mean+3sd':>11}{'임계':>9}   비고")
    print("  " + "-" * 84)
    for c, r in out.iterrows():
        if r["하한적용"]:
            note = "하한 적용"
        elif r["임계"] > r["mean_3sd"] * 1.3:
            note = "★ 단일 점프 지배"
        else:
            note = ""
        print(f"  {c:<32}{r['m6_결측률']:>8.2f}%{r['평상시최대']:>10.2f}p"
              f"{r['mean_3sd']:>10.2f}p{r['임계']:>8.2f}p   {note}")

    n_floor = int(out["하한적용"].sum())
    print(f"\n  하한이 적용된 컬럼 {n_floor}/{len(features)}개")
    print(f"  결측이 발생하는 컬럼(m6 > 0%): "
          f"{int((out['m6_결측률'] > 0).sum())}개")

    if args.no_mlflow:
        print("  [skip] --no-mlflow")
        return

    payload = {
        "multiplier": args.multiplier,
        "floor_pp": args.floor,
        "months": OBSERVED_MONTHS,
        "bounds_pp": {c: float(bound[c]) for c in features},
        "normal_max_pp": {c: float(normal_max[c]) for c in features},
        "mean_plus_3sd_pp": {c: float(mean_3sd[c]) for c in features},
        "m6_missing_pct": {c: float(rate[VALID_MONTHS[0]][c]) for c in features},
    }

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(
        model_loader.MODEL_NAME, model_loader.MODEL_ALIAS
    )
    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_dict(payload, "missing_rate_bounds.json")
    print(f"  logged -> runs:/{mv.run_id}  (missing_rate_bounds.json)")


if __name__ == "__main__":
    main()
