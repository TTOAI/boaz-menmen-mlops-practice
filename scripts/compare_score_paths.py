"""A 경로와 C 경로의 점수 대조.

결정 1(점수 분포 기준을 어느 경로로 만들 것인가)에 필요한 정보를 만든다.

  A 경로: runs:/{run_id}/model     + parquet의 category dtype 그대로  (evaluate.py와 동일)
  C 경로: models:/{name}@{alias}   + categories.json 레벨 재적용      (서빙 아티팩트)

두 점수는 이론상 완전히 같아야 한다. 다르면 아티팩트 체인이 깨진 것이다.
alias가 다른 run을 가리키거나, parquet 왕복에서 category가 풀렸거나,
categories.json의 레벨 순서가 학습 시점과 어긋난 경우가 해당한다.

대조 범위는 모델 아티팩트·레지스트리 alias·카테고리 레벨까지다.
결측 규칙은 parquet에 이미 반영되어 있어 이 대조에 포함되지 않고,
행 단위 prepare()와 HTTP 계층도 포함되지 않는다. 그쪽은 B 경로를 소수 표본으로
따로 대조해야 한다.

사용법:
    MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/compare_score_paths.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_loader  # noqa: E402
from preprocess import CATEGORICAL_COLS  # noqa: E402
from train import align_categories, load_splits, split_xy  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
RUN_ID_FILE = ROOT / "models" / "latest_run.json"


def score_path_a(X: pd.DataFrame, run_id: str) -> np.ndarray:
    """학습 경로. evaluate.py가 지표를 계산할 때와 같은 방식."""
    booster = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    return booster.predict(X)


def score_path_c(X: pd.DataFrame, loaded: model_loader.LoadedModel) -> np.ndarray:
    """서빙 경로의 배치판.

    서빙이 요청 1건에 하는 것과 같은 레벨 재적용·컬럼 재정렬을 프레임 전체에 적용한다.
    prepare()를 그대로 쓰지 않는 이유는 그쪽이 행 단위로만 동작하기 때문이다.
    """
    X = X.copy()
    for col in CATEGORICAL_COLS:
        if col not in X.columns:
            continue
        X[col] = pd.Categorical(
            X[col].astype("object"), categories=loaded.categories[col]
        )
    return loaded.booster.predict(X[loaded.feature_names])


def report(a: np.ndarray, c: np.ndarray, threshold: float) -> None:
    diff = np.abs(a - c)
    n_diff = int((diff > 0).sum())

    print(f"  행 수          {len(a):,}")
    print(f"  최대 절대차     {diff.max():.3e}")
    print(f"  차이 있는 행    {n_diff:,} ({n_diff / len(a) * 100:.4f}%)")
    print(f"  상관계수        {np.corrcoef(a, c)[0, 1]:.10f}")

    # 점수 차이보다 이쪽이 실질적 영향이다. 판정이 바뀌지 않으면 운영상 동일하다.
    alert_a, alert_c = a >= threshold, c >= threshold
    flipped = int((alert_a != alert_c).sum())

    print(f"  임계값          {threshold:.6f}")
    print(f"  경보율 A        {alert_a.mean() * 100:.4f}%  ({int(alert_a.sum()):,}건)")
    print(f"  경보율 C        {alert_c.mean() * 100:.4f}%  ({int(alert_c.sum()):,}건)")
    print(f"  판정 뒤집힘     {flipped:,}건")

    print()
    if n_diff == 0:
        print("  => 두 경로가 완전히 일치. 아티팩트 체인 건전.")
    elif flipped == 0:
        print("  => 점수는 갈리지만 판정은 동일. 원인 확인 필요.")
    else:
        print("  => 판정이 갈림. 기준 분포는 서빙 경로로 만들어야 한다.")


def main() -> None:
    print(f"tracking uri : {mlflow.get_tracking_uri()}")

    print("[1/4] load parquet")
    train_df, valid_df = load_splits(DATA_DIR)
    train_df, valid_df = align_categories(train_df, valid_df)
    X_valid, _ = split_xy(valid_df)

    print("[2/4] resolve runs")
    run_id_a = json.loads(RUN_ID_FILE.read_text())["run_id"]

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(
        model_loader.MODEL_NAME, model_loader.MODEL_ALIAS
    )

    print(f"  A  runs:/{run_id_a}")
    print(f"  C  {model_loader.MODEL_NAME}@{model_loader.MODEL_ALIAS} "
          f"v{mv.version} -> runs:/{mv.run_id}")
    if run_id_a != mv.run_id:
        print("  [warn] 두 경로가 서로 다른 run을 가리킵니다. "
              "아래 대조는 모델 자체가 다른 상태의 비교입니다.")

    print("[3/4] score")
    a = score_path_a(X_valid, run_id_a)
    loaded = model_loader.load()
    c = score_path_c(X_valid, loaded)

    print("[4/4] compare")
    report(a, c, loaded.threshold)


if __name__ == "__main__":
    main()
