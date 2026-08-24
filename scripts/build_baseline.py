"""month 6 기준 분포 생성.

7주차 모니터링이 month 7과 대조할 기준선을 만든다.

  계산: C 경로(레지스트리 모델 + categories.json 배치 예측)
        compare_score_paths.py / compare_serving_path.py에서 서빙 경로와
        점수가 완전히 일치함을 측정했으므로 배치로 만들어도 내용이 같다.
  본체: logs/baseline_m6.jsonl — predict_log와 같은 6필드
  요약: 경보율과 분위수를 모델 run에 metric으로 기록

구간 라벨은 파일명과 application_id 접두사 양쪽에 남긴다.
파일이 합쳐져도 접두사로 복구된다.

scored_at 주의. 이 필드는 **리플레이를 실행한 시각**이며 원본 데이터의 시간이
아니다. BAF Base의 시간축은 month 하나뿐이라 넣을 다른 시각이 없다.
배치 전체가 같은 값을 갖는다.

요약 지표를 새 run이 아니라 모델 run에 붙이는 이유. 기준선은 그 모델의
속성이며 모델이 바뀌면 기준선도 바뀐다. alias로 모델을 찾으면 기준선이
따라오게 되고, 재학습 이후에는 새 모델의 run에 새 기준선이 붙는다.
evaluate.py가 학습 run에 평가 지표를 붙이는 것과 같은 패턴이다.

month 7은 아직 홀드아웃이고 preprocess.py가 저장하지도 않으므로 이 스크립트는
month 6 전용이다. 7주차에 확장한다.

사전 조건:
    docker compose up -d mlflow

사용법:
    MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/build_baseline.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_loader  # noqa: E402
from predict_log import FIELDS  # noqa: E402
from train import align_categories, load_splits, split_xy  # noqa: E402

# C 경로를 재구현하지 않는다. 레벨 재적용 규약이 두 곳에 생기면 안 된다.
from _preflight import require_mlflow  # noqa: E402
from compare_score_paths import score_path_c  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
PREFIX = "m6"
PERCENTILES = (50, 90, 95, 99)


def write_log(path: Path, scores: np.ndarray, loaded, scored_at: str) -> None:
    """6필드 jsonl 배치 쓰기.

    predict_log.write()를 행마다 호출하지 않는 이유는 그쪽이 요청 1건마다
    파일을 열고 닫기 때문이다. 필드명과 순서는 FIELDS에서 그대로 가져온다.

    덮어쓰기(append 아님)다. 기준 분포는 재실행해도 같은 결과여야 한다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        n = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"  [info] {path} 가 이미 있습니다 ({n:,}줄). 덮어씁니다.")

    threshold = loaded.threshold
    with path.open("w", encoding="utf-8") as f:
        for i, score in enumerate(scores):
            record = {
                "application_id": f"{PREFIX}-{i:06d}",
                "score": float(score),
                # serve.py의 판정 규칙과 동일하게 유지할 것
                "decision": "ALERT" if score >= threshold else "PASS",
                "model_version": loaded.version,
                "threshold": threshold,
                "scored_at": scored_at,
            }
            assert tuple(record) == FIELDS
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(scores: np.ndarray, threshold: float) -> dict[str, float]:
    alerts = scores >= threshold
    summary = {
        "baseline_n": float(len(scores)),
        "baseline_alert_rate": float(alerts.mean()),
        "baseline_score_mean": float(scores.mean()),
    }
    for p, v in zip(PERCENTILES, np.percentile(scores, PERCENTILES)):
        summary[f"baseline_score_p{p}"] = float(v)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="month 6 기준 분포 생성")
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "baseline_m6.jsonl")
    parser.add_argument("--no-mlflow", action="store_true", help="요약 지표 기록 생략")
    args = parser.parse_args()

    print(f"tracking uri : {mlflow.get_tracking_uri()}")
    require_mlflow()

    print("[1/5] load parquet")
    train_df, valid_df = load_splits(DATA_DIR)
    _, valid_df = align_categories(train_df, valid_df)
    X_valid, _ = split_xy(valid_df)
    print(f"  X_valid {X_valid.shape}")

    print("[2/5] load model")
    loaded = model_loader.load()
    print(f"  {model_loader.MODEL_NAME}@{model_loader.MODEL_ALIAS} "
          f"v{loaded.version}  threshold {loaded.threshold:.6f}")

    print("[3/5] score (C 경로)")
    scores = score_path_c(X_valid, loaded)

    print("[4/5] write log")
    scored_at = datetime.now(timezone.utc).isoformat()
    write_log(args.out, scores, loaded, scored_at)
    print(f"  {args.out}  {len(scores):,}줄")
    print(f"  scored_at = {scored_at}  (리플레이 시각. 원본 데이터의 시간이 아님)")

    print("[5/5] summarize")
    summary = summarize(scores, loaded.threshold)
    for k, v in summary.items():
        print(f"  {k:<26} {v:.6f}")

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
