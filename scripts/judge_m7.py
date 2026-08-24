"""month 7 2차 판정 — 라벨 사용.

docs/drift-response.md 3절의 절차를 실행한다.
scripts/observe_m7.py 가 만든 점수에 라벨을 붙여 판정한다.

**이번 회차는 판정이 아니라 검증이다.**
1차에서 PSI 0.01283 < 0.046436 으로 정상 판정이 났으므로, 문서 절차대로면
2차를 수행하지 않는다. 그럼에도 수행하는 이유는 두 가지다.

  1. 로드맵 3단계 필수 요건이 "홀드아웃 해제 구간에서 성능 재측정, 기준선
     대비 변화 확인"을 요구한다.
  2. 라벨 없는 1차 신호가 옳았는지를 라벨로 확인할 수 있다. 드리프트가
     있을 때 잡았는지보다, **없을 때 없다고 말했는지**가 덜 검증된다.

성능 재측정을 두 방식으로 한다. 이 둘이 갈리면 그 자체가 진단이다.

  고정 임계 적용   threshold 0.041272 을 그대로 써서 잰 TPR/FPR. 운영 실적.
  FPR 5% 재조정    month 7 에서 예산을 다시 맞췄을 때의 TPR. 모델의 잠재 능력.

  전자만 하락      -> 컷오프가 어긋난 것 (입력 분포 변화)
  둘 다 하락       -> 모델 능력 저하 (관계 변화)

사전 조건:
    python scripts/observe_m7.py
    docker compose up -d mlflow

사용법:
    MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/judge_m7.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_loader  # noqa: E402
from evaluate import evaluate  # noqa: E402
from preprocess import TARGET  # noqa: E402

from _preflight import require_mlflow  # noqa: E402

OBSERVED = ROOT / "logs" / "observed_m7.jsonl"
BASELINE = ROOT / "logs" / "baseline_m6.jsonl"
HOLDOUT = ROOT / "data" / "processed" / "holdout.parquet"

# docs/drift-response.md 1절, 3절
BASE_TPR = 0.503448
BASE_AUPRC = 0.154113


def load_reference(run_id: str) -> dict:
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="drift_reference.json"
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scores_from(path: Path) -> tuple[np.ndarray, float]:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return np.array([r["score"] for r in recs]), float(recs[0]["threshold"])


def at_fixed_threshold(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    pred = s >= t
    pos, neg = y == 1, y == 0
    return {
        "tpr": float(pred[pos].mean()),
        "fpr": float(pred[neg].mean()),
        "alert_rate": float(pred.mean()),
    }


def main() -> None:
    argparse.ArgumentParser(description="month 7 2차 판정").parse_args()

    print(f"tracking uri : {mlflow.get_tracking_uri()}")
    require_mlflow()

    print("[1/4] load")
    s7, threshold = scores_from(OBSERVED)
    s6, _ = scores_from(BASELINE)
    y7 = pd.read_parquet(HOLDOUT, engine="pyarrow", columns=[TARGET])[TARGET].to_numpy()
    if len(y7) != len(s7):
        raise ValueError(f"행 수 불일치: 점수 {len(s7):,} vs 라벨 {len(y7):,}")
    print(f"  m7 {len(s7):,}행 / 사기 {int(y7.sum()):,}건 ({y7.mean()*100:.3f}%)")

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(
        model_loader.MODEL_NAME, model_loader.MODEL_ALIAS
    )
    ref = load_reference(mv.run_id)

    print("[2/4] 성능 재측정 (로드맵 필수 요건)")
    recal = evaluate(y7, s7)
    fixed = at_fixed_threshold(y7, s7, threshold)

    print(f"\n  주지표 TPR@FPR5%")
    print(f"    기준선 (m6)          {BASE_TPR:.6f}")
    print(f"    m7 FPR5% 재조정      {recal['tpr_at_5pct_fpr']:.6f}"
          f"   ({(recal['tpr_at_5pct_fpr']-BASE_TPR)/BASE_TPR*100:+.2f}%)")
    print(f"    m7 고정임계 적용     {fixed['tpr']:.6f}"
          f"   ({(fixed['tpr']-BASE_TPR)/BASE_TPR*100:+.2f}%)")
    print(f"    m7 실제 FPR          {fixed['fpr']:.6f}"
          f"   (예산 0.05 {'초과' if fixed['fpr'] > 0.05 else '이내'})")
    print(f"    m7 재조정 임계값      {recal['threshold_at_5pct_fpr']:.6f}"
          f"   (현재 {threshold:.6f})")

    print("\n[3/4] 2차 판정 지표")
    auprc = recal["auprc"]
    lo_a = ref["auprc"]["lo"]
    a_bad = auprc < lo_a
    print(f"\n  AUPRC   {BASE_AUPRC:.6f} → {auprc:.6f}   하한 {lo_a:.6f}   "
          f"{'★ 저하' if a_bad else '유지'}")

    edges = np.array([-np.inf, *ref["bin_inner_edges"], np.inf])
    b6, b7 = np.digitize(s6, edges[1:-1]), np.digitize(s7, edges[1:-1])
    print(f"\n  {'구간':<20}{'m6 기준':>9}{'m7':>9}{'신뢰구간':>20}{'':>4}{'m7 인원':>9}")
    print("  " + "-" * 74)

    below, above = [], []
    for i, f in enumerate(ref["fraud_rate"]):
        m = b7 == i
        r7 = float(y7[m].mean()) if m.any() else float("nan")
        lo, hi = f["lo"], f["hi"]
        mark = "   "
        if r7 < lo:
            below.append(f["label"])
            mark = " ▼ "
        elif r7 > hi:
            above.append(f["label"])
            mark = " ▲ "
        n6, n7 = int((b6 == i).sum()), int(m.sum())
        print(f"  {f['label']:<20}{f['point']*100:>8.3f}%{r7*100:>8.3f}%"
              f"   [{lo*100:>6.3f}, {hi*100:>6.3f}]{mark}"
              f"{n7:>8,} ({n7-n6:+,})")

    # p0-50 은 신뢰구간 폭이 55% 라 단독 이탈은 무시한다 (문서 3.2, 한계 4번).
    weak = ref["fraud_rate"][0]["label"]
    below = below if any(l != weak for l in below) else []
    above = above if any(l != weak for l in above) else []

    print("\n[4/4] 결론  (문서 3.2)")
    print(f"\n  1차 신호   PSI 0.01283 < 0.046436  →  정상")
    print(f"  AUPRC      {'하한 미만' if a_bad else '구간 내 또는 상방'}")
    print(f"  구간 이탈   하방 {below or '없음'} / 상방 {above or '없음'}")

    print()
    if below and a_bad:
        print("  => 관계 변화(악화). 재학습 대상.")
        print("     1차가 정상이었다면 PSI 임계·지표 선택을 재검토할 것.")
    elif below:
        print("  => 판단 보류. 하방 이탈이 있으나 AUPRC 는 하한 이상이다.")
        print("     lab note 에 기록하고 추가 관측을 기다린다.")
    elif above:
        print("  => 관계 변화(유리). 조치 없음, 관측 지속. (문서 3.2.1)")
        print("     같은 점수 구간의 실제 사기율이 올랐다. 모델이 실제보다")
        print("     보수적으로 점수를 매기고 있으며 고정 임계 성능이 개선됐다.")
        over_budget = fixed["fpr"] > 0.05
        print(f"     실제 FPR {fixed['fpr']:.6f} — 예산 0.05 "
              f"{'초과. 임계값 재보정 필요.' if over_budget else '이내. 임계값 유지.'}")
        print("     다음 회차에 방향이 뒤집힐 수 있으므로 원인을 기록할 것.")
    else:
        print("  => 성능 유지. 1차의 정상 판정이 라벨로도 확인됐다.")
        print("     라벨 없는 신호가 라벨 있는 진실과 일치했다.")

    metrics = {
        "m7_auprc": auprc,
        "m7_tpr_at_5pct_fpr": recal["tpr_at_5pct_fpr"],
        "m7_tpr_at_fixed_threshold": fixed["tpr"],
        "m7_fpr_at_fixed_threshold": fixed["fpr"],
        "m7_alert_rate": fixed["alert_rate"],
        "m7_threshold_recalibrated": recal["threshold_at_5pct_fpr"],
    }
    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_metrics(metrics)
    print(f"\n  logged -> runs:/{mv.run_id}")


if __name__ == "__main__":
    main()
