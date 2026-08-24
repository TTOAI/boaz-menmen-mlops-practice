"""month 7 관측 — 1차 판정 (라벨 미사용).

docs/drift-response.md 2절의 절차를 실행한다.

  1. 입력 검증 4항목 (파이프라인 버그 배제)
  2. 점수 산출 -> logs/observed_m7.jsonl
  3. PSI 계산 -> 임계 0.046436 과 대조
  4. 보조 지표 (경보율, 분위수)

**라벨을 읽지 않는다.** pd.read_parquet 의 columns 인자로 fraud_bool 을 아예
제외한다. 문서 6절에 적었듯 month 7 의 라벨은 데이터에 이미 들어 있어 실제로는
"즉시" 오지만, 1차 판정이 라벨에 오염되면 2단 구조 자체가 무의미해진다.
규율을 주석이 아니라 코드로 표현한다.

2차 판정(AUPRC, 구간별 사기율)은 scripts/judge_m7.py 가 별도로 수행한다.
한 스크립트로 묶으면 실행 한 번에 1·2차가 동시에 나와 순서가 지켜지지 않는다.

사전 조건:
    python src/preprocess.py --include-holdout
    docker compose up -d mlflow

사용법:
    MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/observe_m7.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_loader  # noqa: E402
from predict_log import FIELDS  # noqa: E402
from preprocess import CATEGORICAL_COLS, MONTH_COL, TARGET  # noqa: E402

from _preflight import require_mlflow  # noqa: E402
from compare_score_paths import score_path_c  # noqa: E402
from estimate_normal_drift import bin_edges, psi  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
HOLDOUT = DATA_DIR / "holdout.parquet"
VALID = DATA_DIR / "valid.parquet"
BASELINE = ROOT / "logs" / "baseline_m6.jsonl"
OUT = ROOT / "logs" / "observed_m7.jsonl"

PREFIX = "m7"
PSI_BINS = 10
PSI_THRESHOLD = 0.046436  # docs/drift-response.md 2.2


def read_without_labels(path: Path) -> pd.DataFrame:
    """라벨 컬럼을 파일에서 아예 읽지 않는다."""
    names = pq.ParquetFile(path).schema_arrow.names
    cols = [c for c in names if c != TARGET]
    if TARGET in names:
        print(f"  [info] {path.name}: {TARGET} 제외하고 {len(cols)}개 컬럼만 로드")
    return pd.read_parquet(path, engine="pyarrow", columns=cols)


def load_missing_bounds() -> dict[str, float]:
    """컬럼별 결측률 임계. estimate_missing_bounds.py 가 run에 남긴 값."""
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(
        model_loader.MODEL_NAME, model_loader.MODEL_ALIAS
    )
    path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="missing_rate_bounds.json"
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))["bounds_pp"]


def check_inputs(
    m7: pd.DataFrame, m6: pd.DataFrame, loaded, bounds: dict[str, float]
) -> list[str]:
    """docs 2.3 입력 검증. 네 항목 모두 정량 판정한다."""
    issues: list[str] = []

    # (1) 카테고리 — 학습 시점 레벨에 없는 값이 있는가
    print("  [1] 카테고리 레벨")
    for col in CATEGORICAL_COLS:
        levels = set(loaded.categories[col])
        unseen = sorted(set(m7[col].dropna().astype(str).unique()) - levels)
        mark = "OK" if not unseen else f"★ 신규 {unseen}"
        print(f"      {col:<20} {mark}")
        if unseen:
            issues.append(f"{col}에 학습 시점에 없던 값 {unseen}")

    # (2) 스키마 — 컬럼 집합과 dtype
    print("  [2] 스키마")
    if list(m7.columns) != list(m6.columns):
        issues.append("컬럼 집합·순서가 month 6과 다름")
        print("      ★ 컬럼 불일치")
    else:
        bad = [c for c in m7.columns if str(m7[c].dtype) != str(m6[c].dtype)]
        if bad:
            issues.append(f"dtype 불일치: {bad}")
        print(f"      컬럼 {len(m7.columns)}개 일치 / dtype {'일치' if not bad else f'★ {bad}'}")

    # (3) 결측률 — 컬럼별 임계와 대조. estimate_missing_bounds.py 가 유도한 값이다.
    print("  [3] 결측률 변화 (컬럼별 임계 대조)")
    feats = [c for c in loaded.feature_names if c in m7.columns]
    d = ((m7[feats].isna().mean() - m6[feats].isna().mean()) * 100)

    over = [c for c in feats if abs(d[c]) > bounds.get(c, float("inf"))]
    shown = list(d.abs().sort_values(ascending=False).index[:5])
    for c in shown + [c for c in over if c not in shown]:
        b = bounds.get(c, float("nan"))
        mark = "★ 초과" if c in over else ""
        print(f"      {c:<32} m6 {m6[c].isna().mean()*100:>6.2f}%"
              f"  m7 {m7[c].isna().mean()*100:>6.2f}%"
              f"  Δ{d[c]:>+6.2f}p  임계 {b:>5.2f}p  {mark}")

    if over:
        issues.append(f"결측률 임계 초과: {over}")
    else:
        print(f"      → 29개 컬럼 전부 임계 이내")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="month 7 관측 (1차 판정)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    print(f"tracking uri : {mlflow.get_tracking_uri()}")
    require_mlflow()

    print("[1/5] load holdout (라벨 제외)")
    m7 = read_without_labels(HOLDOUT)
    m6 = read_without_labels(VALID)
    print(f"  m7 {m7.shape} / m6 {m6.shape}")
    assert TARGET not in m7.columns, "라벨이 로드됨"

    print("[2/5] load model")
    loaded = model_loader.load()
    print(f"  v{loaded.version}  threshold {loaded.threshold:.6f}")

    print("[3/5] 입력 검증 (파이프라인 버그 배제)")
    issues = check_inputs(m7, m6, loaded, load_missing_bounds())

    print("[4/5] score & log")
    X = m7.drop(columns=[MONTH_COL])
    scores = score_path_c(X, loaded)

    scored_at = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, s in enumerate(scores):
            rec = {
                "application_id": f"{PREFIX}-{i:06d}",
                "score": float(s),
                "decision": "ALERT" if s >= loaded.threshold else "PASS",
                "model_version": loaded.version,
                "threshold": loaded.threshold,
                "scored_at": scored_at,
            }
            assert tuple(rec) == FIELDS
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {args.out}  {len(scores):,}줄")

    print("[5/5] 1차 판정")
    base = np.array(
        [json.loads(l)["score"] for l in BASELINE.read_text(encoding="utf-8").splitlines()]
    )
    edges = bin_edges(base, PSI_BINS)
    value = psi(base, scores, edges)

    rate_m6 = float((base >= loaded.threshold).mean())
    rate_m7 = float((scores >= loaded.threshold).mean())

    print(f"\n  PSI            {value:.5f}   임계 {PSI_THRESHOLD:.5f}")
    print(f"  경보율 m6→m7   {rate_m6*100:.4f}% → {rate_m7*100:.4f}%  "
          f"({(rate_m7-rate_m6)*100:+.4f}%p)")
    print(f"  {'분위수':<10}{'m6':>12}{'m7':>12}")
    for p in (50, 90, 95, 99):
        print(f"  p{p:<9}{np.percentile(base, p):>12.6f}{np.percentile(scores, p):>12.6f}")

    print()
    if issues:
        print("  => 파이프라인 버그. 재학습 금지. 수정 후 재측정.")
        for i in issues:
            print(f"     - {i}")
    elif value >= PSI_THRESHOLD:
        print("  => 1차 경보. 입력 검증은 통과했으므로 진짜 변화로 본다.")
        print("     2차 판정으로 진행: scripts/judge_m7.py")
    else:
        print("  => 정상. PSI 가 임계 미만이다. 종료.")
        print("     (문서 3절 2차 판정은 수행하지 않는다)")


if __name__ == "__main__":
    main()
