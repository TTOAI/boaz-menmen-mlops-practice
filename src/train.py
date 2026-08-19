"""
BAF Base 학습 스크립트

data/processed/{train,valid}.parquet -> LightGBM 학습 -> MLflow run 기록

공통:
  - LightGBM 하이퍼파라미터는 기본값 고정
  - random_state = 42
  - month는 피처에서 제외 (학습 0~5 / 검증 6으로 구간이 분리되어 있어
    피처로 넣으면 모델이 month로 분기해버린다)
  - 범주형 5개는 category dtype 그대로 LightGBM 네이티브 처리

사용법:
    python src/train.py
    python src/train.py --data data/processed --experiment baf-baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from preprocess import (
    CATEGORICAL_COLS,
    MONTH_COL,
    TARGET,
    TRAIN_MONTHS,
    VALID_MONTHS,
)

# month는 피처가 아니다. 분할 기준일 뿐이다.
NON_FEATURE_COLS = [TARGET, MONTH_COL]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전처리된 parquet 로드."""
    paths = {name: data_dir / f"{name}.parquet" for name in ("train", "valid")}
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"{p} 가 없습니다. 먼저 `python src/preprocess.py` 를 실행하세요."
            )
    train = pd.read_parquet(paths["train"], engine="pyarrow")
    valid = pd.read_parquet(paths["valid"], engine="pyarrow")
    return train, valid


def align_categories(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """category dtype 보존 여부를 확인하고 train 기준으로 정렬한다.

    pyarrow 버전에 따라 parquet 왕복에서 category가 일반 object로 풀릴 수 있다.
    또 train/valid의 카테고리 집합이 다르면 LightGBM 내부 코드가 어긋나므로
    valid를 train의 dtype에 맞춘다.
    """
    train, valid = train.copy(), valid.copy()

    for col in CATEGORICAL_COLS:
        if col not in train.columns:
            continue
        if not isinstance(train[col].dtype, pd.CategoricalDtype):
            print(f"[warn] {col} 이 category가 아닙니다({train[col].dtype}). 재캐스팅합니다.")
            train[col] = train[col].astype("category")
        # valid를 train의 카테고리 집합에 강제로 맞춘다
        valid[col] = valid[col].astype(train[col].dtype)

    return train, valid


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df[features], df[TARGET]


def build_model() -> lgb.LGBMClassifier:
    """기본 하이퍼파라미터 + random_state만 고정."""
    return lgb.LGBMClassifier(random_state=RANDOM_STATE)


def split_config() -> dict:
    return {
        "train_months": ",".join(map(str, TRAIN_MONTHS)),
        "valid_months": ",".join(map(str, VALID_MONTHS)),
        "split_type": "temporal",
    }


# ---------------------------------------------------------------------------


def run(data_dir: Path, experiment: str, run_id_path: Path) -> str:
    print("[1/4] load parquet")
    train_df, valid_df = load_splits(data_dir)
    train_df, valid_df = align_categories(train_df, valid_df)

    X_train, y_train = split_xy(train_df)
    X_valid, y_valid = split_xy(valid_df)

    print(f"  train {X_train.shape} / fraud {int(y_train.sum()):,}")
    print(f"  valid {X_valid.shape} / fraud {int(y_valid.sum()):,}")
    print(f"  features {X_train.shape[1]}개 (month 제외)")

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    print(f"  categorical {cat_cols}")

    print("[2/4] start mlflow run")
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as active:
        run_id = active.info.run_id

        model = build_model()

        # 고정 params
        mlflow.log_params(model.get_params())
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_params(split_config())
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("categorical_cols", ",".join(cat_cols))
        mlflow.log_param("early_stopping", False)

        print("[3/4] fit")
        model.fit(X_train, y_train, categorical_feature=cat_cols)

        print("[4/4] log model")
        mlflow.lightgbm.log_model(model.booster_, artifact_path="model")

        # evaluate.py가 같은 run에 지표를 붙일 수 있도록 run_id를 남긴다
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(json.dumps({"run_id": run_id, "experiment": experiment}))

        print(f"  run_id = {run_id}")
        print(f"  saved  -> {run_id_path}")

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="BAF LightGBM 베이스라인 학습")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--experiment", type=str, default="baf-baseline")
    parser.add_argument("--run-id-file", type=Path, default=Path("models/latest_run.json"))
    args = parser.parse_args()

    run(args.data, args.experiment, args.run_id_file)


if __name__ == "__main__":
    main()