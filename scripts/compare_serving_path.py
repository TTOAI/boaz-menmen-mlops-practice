"""B 경로(서빙 API)와 C 경로(배치)의 점수 대조 — 표본.

compare_score_paths.py가 닫지 못한 두 층을 잰다.
  - 행 단위 prepare(): 결측 규칙 재적용, dtype 강제 변환, 컬럼 재정렬
  - HTTP / pydantic 직렬화

입력은 raw CSV에서 뽑는다. parquet를 쓰면 결측 규칙이 이미 반영되어 있어
apply_missing_rules가 no-op이 되고 정작 재려던 층이 빠진다.
서빙의 입력 계약도 raw이므로 이쪽이 실제 운영과 같다.

행 매칭 근거. preprocess.py의 변환(drop/mask/astype)은 전부 행 순서를 보존하고
split_by_month가 reset_index(drop=True)를 하므로, raw에서 month 6을 같은 순서로
뽑으면 valid.parquet와 위치가 1:1 대응한다.

사전 조건 2가지.
  1. MLflow 서버
       docker compose up -d mlflow
  2. 서빙 API. 예측 로그를 반드시 분리해서 띄울 것.
     기본 경로로 띄우면 결정 1의 기준 로그에 검증용 호출이 섞인다.
       PREDICT_LOG_PATH=logs/skew_check.jsonl \
       MLFLOW_TRACKING_URI=http://localhost:5050 \
       uvicorn serve:app --app-dir src --port 8000

사용법:
    MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/compare_serving_path.py
    ... --n 500 --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_loader  # noqa: E402
from preprocess import MONTH_COL, VALID_MONTHS  # noqa: E402
from train import align_categories, load_splits, split_xy  # noqa: E402

# C 경로는 재구현하지 않고 그대로 가져온다. 레벨 재적용 규약이 두 곳에 생기면
# 이 스크립트가 재려는 대상 자체가 흐려진다.
from compare_score_paths import score_path_c  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"


def _json_safe(v):
    """numpy 스칼라와 NaN을 JSON이 표현할 수 있는 값으로 바꾼다."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def sample_indices(raw_path: Path, n: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """raw CSV에서 month 6 부분집합을 뽑고, valid.parquet 기준 위치를 함께 반환."""
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} 가 없습니다.")

    print(f"  read {raw_path} (전체 로드, 수 초 소요)")
    raw = pd.read_csv(raw_path)
    m6 = raw[raw[MONTH_COL].isin(VALID_MONTHS)].reset_index(drop=True)
    print(f"  month {VALID_MONTHS} → {len(m6):,}행")

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(m6), size=min(n, len(m6)), replace=False))
    return idx, m6.iloc[idx]


def score_b(rows: pd.DataFrame, url: str, feature_names: list[str]) -> tuple[np.ndarray, list]:
    """요청 1건씩 /predict 호출. 실패는 점수 대신 사유로 모은다."""
    endpoint = url.rstrip("/") + "/predict"
    scores: list[float] = []
    failures: list[tuple[int, str]] = []

    for i, (_, row) in enumerate(rows.iterrows()):
        # feature_names로 거르면 target·month·device_fraud_count가 함께 빠진다.
        features = {k: _json_safe(row[k]) for k in feature_names}
        body = json.dumps(
            {"application_id": f"skewcheck-{i:05d}", "features": features}
        ).encode()
        req = urllib.request.Request(
            endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as res:
                scores.append(float(json.load(res)["score"]))
        except urllib.error.HTTPError as e:
            failures.append((i, f"{e.code} {e.read().decode(errors='replace')[:200]}"))
            scores.append(math.nan)
        except urllib.error.URLError as e:
            raise SystemExit(
                f"서빙 API에 연결하지 못했습니다 ({endpoint}): {e.reason}\n"
                "모듈 docstring의 사전 조건 2를 확인하세요."
            ) from e

        if (i + 1) % 100 == 0:
            print(f"    {i + 1:,}/{len(rows):,}")

    return np.asarray(scores), failures


def report(b: np.ndarray, c: np.ndarray, threshold: float, failures: list) -> None:
    print(f"  요청 수        {len(b):,}")
    print(f"  실패           {len(failures):,}건")
    for i, why in failures[:5]:
        print(f"    [{i}] {why}")
    if len(failures) > 5:
        print(f"    ... 외 {len(failures) - 5:,}건")

    ok = ~np.isnan(b)
    if not ok.any():
        print("\n  => 성공한 요청이 없어 대조 불가.")
        return

    b, c = b[ok], c[ok]
    diff = np.abs(b - c)
    n_diff = int((diff > 0).sum())

    print(f"  대조 대상      {len(b):,}")
    print(f"  최대 절대차     {diff.max():.3e}")
    print(f"  차이 있는 행    {n_diff:,} ({n_diff / len(b) * 100:.4f}%)")

    alert_b, alert_c = b >= threshold, c >= threshold
    flipped = int((alert_b != alert_c).sum())
    print(f"  임계값          {threshold:.6f}")
    print(f"  경보 B / C      {int(alert_b.sum()):,} / {int(alert_c.sum()):,}")
    print(f"  판정 뒤집힘     {flipped:,}건")

    print()
    if failures:
        print("  => 실패한 요청이 있습니다. 대조 이전에 그쪽 원인부터 확인할 것.")
    elif n_diff == 0:
        print("  => 서빙 경로와 배치 경로가 완전히 일치. train-serving skew 없음.")
    elif flipped == 0:
        print("  => 점수는 갈리지만 판정은 동일. 원인 확인 필요.")
    else:
        print("  => 판정이 갈림. 서빙 경로에 결함이 있다. 7주차 이전에 해결할 것.")


def main() -> None:
    parser = argparse.ArgumentParser(description="서빙 경로 vs 배치 경로 표본 대조")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "Base.csv")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[1/4] sample raw rows")
    idx, raw_rows = sample_indices(args.raw, args.n, args.seed)

    print("[2/4] load model & parquet")
    loaded = model_loader.load()
    print(f"  model v{loaded.version}  threshold {loaded.threshold:.6f}")

    train_df, valid_df = load_splits(DATA_DIR)
    _, valid_df = align_categories(train_df, valid_df)
    X_valid, _ = split_xy(valid_df)
    X_sub = X_valid.iloc[idx]

    print(f"[3/4] call {args.url}/predict  ({len(raw_rows):,}건)")
    b, failures = score_b(raw_rows, args.url, loaded.feature_names)

    print("[4/4] compare")
    c = score_path_c(X_sub, loaded)
    report(b, c, loaded.threshold, failures)


if __name__ == "__main__":
    main()
