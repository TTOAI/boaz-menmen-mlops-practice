"""MLflow tracking 서버 연결을 미리 확인한다.

MLflow 클라이언트는 서버가 없을 때 재시도 백오프로 수 분을 소비한 뒤에야
실패한다. 그동안 출력이 없어 원인을 짚기 어렵다. 이 세션에서 두 번 물렸다.

근본 원인은 model_loader.load() 에 타임아웃이 없다는 것이다. serve.py 의
lifespan 도 같은 문제를 갖는다 — 레지스트리가 닿지 않으면 API 가 즉시 실패하지
않고 몇 분간 재시도한 뒤 죽으며, 그동안 컨테이너는 기동 중으로 보인다.
그쪽은 2단계 서빙 코드라 여기서 고치지 않는다. 잔여 과제.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from urllib.parse import urlparse

import mlflow

TIMEOUT = 3.0


def require_mlflow(timeout: float = TIMEOUT) -> None:
    """tracking URI 가 HTTP 계열이면 /health 를 한 번 두드려 본다.

    file/sqlite 백엔드는 연결 대상이 아니므로 통과시킨다.
    """
    uri = mlflow.get_tracking_uri()
    parsed = urlparse(uri)

    if parsed.scheme not in ("http", "https"):
        return

    try:
        with urllib.request.urlopen(
            f"{parsed.scheme}://{parsed.netloc}/health", timeout=timeout
        ) as res:
            if res.status == 200:
                return
            reason = f"HTTP {res.status}"
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)

    raise SystemExit(
        f"MLflow tracking 서버에 연결하지 못했습니다.\n"
        f"  uri    {uri}\n"
        f"  사유   {reason}\n"
        f"  조치   docker compose up -d mlflow\n"
        "확인 없이 진행하면 클라이언트가 수 분간 조용히 재시도한 뒤 실패합니다."
    )
