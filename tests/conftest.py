"""테스트 픽스처.

CI에는 원본 데이터가 없으므로, 실제 학습 모델 대신 동일한 계약을 만족하는
소형 모델을 테스트 시점에 만들어 레지스트리에 등록한다.
검증 대상은 모델 성능이 아니라 요청부터 로그까지의 경로다.

train.py와 동일하게 model.booster_를 기록해 서빙 로딩 경로를 그대로 재현한다.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import CATEGORICAL_COLS  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "sample_request.json"
MODEL_NAME = "baf-fraud-lgbm"
ALIAS = "production"
N_ROWS = 400
SEED = 42


@pytest.fixture(scope="session")
def sample() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _synthetic_frame(sample: dict, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(N_ROWS):
        row = {}
        for col, value in sample.items():
            if col in CATEGORICAL_COLS:
                row[col] = value
            elif isinstance(value, bool):
                row[col] = int(rng.integers(0, 2))
            elif isinstance(value, (int, float)):
                row[col] = float(value) + float(rng.normal(0, 1))
            else:
                row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def registered_model(tmp_path_factory, sample):
    tmp = tmp_path_factory.mktemp("mlflow")
    # 모델 레지스트리는 파일 백엔드를 지원하지 않는다. 테스트도 sqlite를 쓴다.
    uri = f"sqlite:///{tmp / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    os.environ["MLFLOW_TRACKING_URI"] = uri

    # artifact_location을 명시하지 않으면 CWD 기준 ./mlruns 로 떨어져
    # 5주차 기록이 남아 있는 리포의 mlruns/ 를 오염시킨다.
    experiment_id = mlflow.create_experiment(
        "integration", artifact_location=(tmp / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    rng = np.random.default_rng(SEED)
    X = _synthetic_frame(sample, rng)

    # 카테고리 집합은 학습 시점에 확정하고 그대로 기록한다.
    categories: dict[str, list] = {}
    for col in CATEGORICAL_COLS:
        levels = sorted({str(sample[col]), "__OTHER__"})
        X[col] = pd.Categorical(X[col].astype(str), categories=levels)
        categories[col] = levels

    y = rng.integers(0, 2, size=len(X))

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    model = lgb.LGBMClassifier(n_estimators=5, random_state=SEED, verbose=-1)
    model.fit(X, y, categorical_feature=cat_cols)

    booster = model.booster_
    head = X.head(50)
    signature = infer_signature(head, booster.predict(head))

    with mlflow.start_run() as active:
        run_id = active.info.run_id
        mlflow.lightgbm.log_model(
            booster,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )
        mlflow.log_dict(categories, "categories.json")
        # 서빙이 읽는 임계값. evaluate.py와 같은 키를 써야 경로가 재현된다.
        mlflow.log_metric("threshold_at_5pct_fpr", 0.5)

    client = mlflow.MlflowClient()
    version = next(
        mv.version
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
        if mv.run_id == run_id
    )
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    return version


@pytest.fixture()
def client(tmp_path, registered_model):
    os.environ["PREDICT_LOG_PATH"] = str(tmp_path / "predictions.jsonl")

    from fastapi.testclient import TestClient

    import serve

    with TestClient(serve.create_app()) as c:
        yield c
