"""서빙이 사용하는 모델 아티팩트 로딩과 입력 전처리.

핵심 규약
- 임계값을 코드 상수로 두지 않는다. 5주차 run에 기록된 값을 모델과 함께 로드한다.
- 결측 규칙은 preprocess.apply_missing_rules를 그대로 재사용한다. 서빙에서 재정의하지 않는다.
- 카테고리 집합은 학습 시점 값을 그대로 적용한다. 요청 1건에서 추론하지 않는다.

입력 계약은 raw다. 전처리된 값이 아니라 신청서 원본(-1 결측 인코딩 포함)을 받는다.
전처리 책임을 호출자에게 넘기면 학습과 서빙의 규약이 갈린다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mlflow
import pandas as pd

# 전처리 규약의 단일 출처. 여기서 규칙을 다시 쓰면 계약이 두 곳에 생긴다.
from preprocess import CATEGORICAL_COLS, apply_missing_rules

MODEL_NAME = os.getenv("MODEL_NAME", "baf-fraud-lgbm")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
# evaluate.py는 valid 기준 지표를 접두사 없이, train 기준 지표를 train_ 접두사로 기록한다.
# 운영에 적용하는 값은 valid 쪽이다.
THRESHOLD_METRIC = os.getenv("THRESHOLD_METRIC", "threshold_at_5pct_fpr")


@dataclass
class LoadedModel:
    booster: object
    version: str
    threshold: float
    categories: dict[str, list]
    feature_names: list[str]


def _resolve_threshold(metrics: dict[str, float], run_id: str) -> float:
    if THRESHOLD_METRIC in metrics:
        return float(metrics[THRESHOLD_METRIC])

    # 키가 바뀐 경우를 대비한 탐색. train_ 접두사는 학습 구간 값이므로 제외한다.
    candidates = [
        k for k in metrics if "threshold" in k.lower() and not k.startswith("train_")
    ]
    if len(candidates) == 1:
        return float(metrics[candidates[0]])

    raise RuntimeError(
        f"run {run_id}에서 임계값을 찾지 못함. metrics={sorted(metrics)}. "
        "evaluate를 재실행해 임계값을 지표와 같은 run에 기록할 것"
    )


def load() -> LoadedModel:
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(mv.run_id)

    threshold = _resolve_threshold(run.data.metrics, mv.run_id)

    # train.py가 model.booster_를 기록하므로 로드 결과도 Booster다.
    booster = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

    path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="categories.json"
    )
    with open(path, encoding="utf-8") as f:
        categories = json.load(f)

    return LoadedModel(
        booster=booster,
        version=str(mv.version),
        threshold=threshold,
        categories=categories,
        feature_names=list(booster.feature_name()),
    )


def prepare(payload: dict, loaded: LoadedModel) -> pd.DataFrame:
    """요청 1건을 학습 시점과 동일한 형태로 변환.

    학습과 어긋나도 예외가 나지 않는 구간이므로,
    누락 컬럼을 조용히 채우지 않고 명시적으로 거부한다.
    """
    missing = [c for c in loaded.feature_names if c not in payload]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = pd.DataFrame([payload])

    # 수치 컬럼에 null이나 문자열이 섞이면 컬럼이 object dtype이 되고,
    # apply_missing_rules의 부등호 비교와 Booster.predict가 둘 다 실패한다.
    # 여기서도 조용히 채우지 않고 거부한다.
    for col in loaded.feature_names:
        if col in CATEGORICAL_COLS:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any() and payload[col] is not None:
            raise ValueError(f"{col} 값 {payload[col]!r}이 수치가 아님")
        df[col] = coerced

    df = apply_missing_rules(df)

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        levels = loaded.categories[col]
        df[col] = pd.Categorical(df[col].astype("object"), categories=levels)
        if df[col].isna().all() and payload[col] is not None:
            raise ValueError(
                f"{col} 값 {payload[col]!r}이 학습 시점 카테고리에 없음: {levels}"
            )

    # 컬럼 순서까지 학습 시점과 일치시킨다.
    return df[loaded.feature_names]


def score(loaded: LoadedModel, df: pd.DataFrame) -> float:
    """Booster.predict는 이진 분류에서 양성 확률을 그대로 반환한다."""
    return float(loaded.booster.predict(df)[0])
