"""
BAF Base 평가 스크립트

train.py가 남긴 run_id로 모델을 불러와 valid(month 6)에서 지표를 계산하고
같은 MLflow run에 metrics를 추가한다.

공통:
  - 주지표: TPR @ FPR 5%
      FPR <= 0.05 를 만족하는 지점 중 최대 TPR
      "FPR 5% 예산 안에서 잡을 수 있는 최대 사기" 라는 해석
  - 보조지표: AUPRC (average_precision_score)
  - accuracy 금지

사용법:
    python src/evaluate.py
    python src/evaluate.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from preprocess import TARGET
from train import align_categories, load_splits, split_xy

FPR_BUDGET = 0.05


def tpr_at_fpr(y_true, y_score, budget: float = FPR_BUDGET) -> dict:
    """FPR 예산 이내에서 달성 가능한 최대 TPR과 그때의 임계값

    ROC 곡선은 유한한 점들의 집합이라 FPR이 정확히 budget인 지점은
    대개 존재하지 않는다. 보간하지 않고, 예산을 넘지 않는 지점 중
    가장 좋은 것을 택한다(예산 초과가 없으므로 보수적).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    within = fpr <= budget
    if not within.any():
        raise ValueError(f"FPR <= {budget} 인 지점이 없습니다")

    idx = int(np.argmax(tpr[within]))  # within 내 최대 TPR
    sel = np.flatnonzero(within)[idx]

    return {
        "tpr": float(tpr[sel]),
        "fpr": float(fpr[sel]),
        "threshold": float(thresholds[sel]),
    }


def evaluate(y_true, y_score) -> dict:
    at = tpr_at_fpr(y_true, y_score)
    return {
        "tpr_at_5pct_fpr": at["tpr"],
        "auprc": float(average_precision_score(y_true, y_score)),
        # 참고용 (주지표 아님)
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold_at_5pct_fpr": at["threshold"],
        "actual_fpr": at["fpr"],
    }


def load_run_info(run_id: str | None, run_id_file: Path) -> str:
    if run_id:
        return run_id
    if not run_id_file.exists():
        raise FileNotFoundError(
            f"{run_id_file} 가 없습니다. 먼저 `python src/train.py` 를 실행하거나 "
            f"--run-id 로 직접 지정하세요."
        )
    return json.loads(run_id_file.read_text())["run_id"]


def print_metrics(name: str, m: dict) -> None:
    print(f"  [{name}]")
    print(f"    TPR@FPR5%  {m['tpr_at_5pct_fpr']:.4f}  (실제 FPR {m['actual_fpr']:.4f})")
    print(f"    AUPRC      {m['auprc']:.4f}")
    print(f"    ROC-AUC    {m['roc_auc']:.4f}  (참고용)")
    print(f"    threshold  {m['threshold_at_5pct_fpr']:.6f}")


def run(data_dir: Path, run_id: str) -> dict:
    print("[1/4] load data")
    train_df, valid_df = load_splits(data_dir)
    train_df, valid_df = align_categories(train_df, valid_df)

    X_train, y_train = split_xy(train_df)
    X_valid, y_valid = split_xy(valid_df)

    print(f"[2/4] load model  runs:/{run_id}/model")
    booster = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

    print("[3/4] predict & score")
    valid_metrics = evaluate(y_valid, booster.predict(X_valid))
    train_metrics = evaluate(y_train, booster.predict(X_train))

    print_metrics("valid (month 6)", valid_metrics)
    print_metrics("train (month 0-5, 참고)", train_metrics)

    gap = train_metrics["tpr_at_5pct_fpr"] - valid_metrics["tpr_at_5pct_fpr"]
    print(f"  train-valid TPR 차이: {gap:+.4f}")

    print("[4/4] log metrics to mlflow")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(valid_metrics)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metric("train_valid_tpr_gap", gap)

    return valid_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="BAF 베이스라인 평가")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-id-file", type=Path, default=Path("models/latest_run.json"))
    args = parser.parse_args()

    run_id = load_run_info(args.run_id, args.run_id_file)
    run(args.data, run_id)


if __name__ == "__main__":
    main()