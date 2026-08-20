"""통합 테스트.

단위 테스트가 각 함수 내부를 보는 것과 달리, 여기서는 단계와 단계 사이를 본다.
요청 1건이 판정을 거쳐 로그 1줄에 도달하는지 확인한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import predict_log


def _log_lines() -> list[dict]:
    path = Path(os.environ["PREDICT_LOG_PATH"])
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_request_reaches_log(client, sample):
    before = len(_log_lines())

    res = client.post("/predict", json={"application_id": "test-1", "features": sample})

    assert res.status_code == 200
    body = res.json()
    assert body["decision"] in ("ALERT", "PASS")
    assert 0.0 <= body["score"] <= 1.0

    lines = _log_lines()
    assert len(lines) == before + 1

    record = lines[-1]
    # 필드명은 스터디 공통 규약이므로 이름까지 검증한다.
    assert tuple(record) == predict_log.FIELDS
    assert record["application_id"] == "test-1"
    assert record["decision"] == body["decision"]
    assert record["model_version"] == body["model_version"]
    assert record["threshold"] == body["threshold"]


def test_decision_follows_loaded_threshold(client, sample):
    res = client.post("/predict", json={"features": sample})
    body = res.json()

    expected = "ALERT" if body["score"] >= body["threshold"] else "PASS"
    assert body["decision"] == expected

    # 임계값이 코드 상수가 아니라 run에서 로드된 값인지 확인
    health = client.get("/health").json()
    assert health["threshold"] == body["threshold"]


def test_missing_column_is_rejected(client, sample):
    broken = dict(sample)
    broken.pop(next(iter(broken)))

    res = client.post("/predict", json={"features": broken})

    # 조용히 채우지 않고 거부되어야 한다.
    assert res.status_code == 422
