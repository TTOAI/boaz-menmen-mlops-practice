"""예측 로그 적재.

7주차 모니터링이 관측할 수 있는 유일한 데이터이며, 소급 생성이 불가능하다.
필드명은 스터디 공통 규약이므로 변경하지 않는다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

FIELDS = (
    "application_id",
    "score",
    "decision",
    "model_version",
    "threshold",
    "scored_at",
)

_lock = Lock()


def log_path() -> Path:
    # 테스트에서 tmp_path로 갈아끼울 수 있도록 환경변수로 분리
    return Path(os.getenv("PREDICT_LOG_PATH", "logs/predictions.jsonl"))


def write(
    application_id: str,
    score: float,
    decision: str,
    model_version: str,
    threshold: float,
) -> dict:
    record = {
        "application_id": application_id,
        "score": score,
        "decision": decision,
        "model_version": model_version,
        "threshold": threshold,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    assert tuple(record) == FIELDS

    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return record
