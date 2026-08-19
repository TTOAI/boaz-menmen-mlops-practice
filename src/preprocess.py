"""
BAF Base 전처리 스크립트

data/raw/Base.csv -> data/processed/{train,valid}.parquet

공통:
  - device_fraud_count 드롭 (전체 값이 0, 정보량 없음)
  - == -1 을 NaN 처리하는 컬럼 5개
  - intended_balcon_amount 는 < 0 을 NaN 처리 (정확히 -1인 값이 0건)
  - credit_risk_score, velocity_6h 의 음수는 정상값이므로 유지
  - 범주형 5개는 pandas category dtype (LightGBM 네이티브 처리용)
  - month 0~5 = train / month 6 = valid / month 7 = 드리프트 실험 전용(기본 제외)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


TARGET = "fraud_bool"
MONTH_COL = "month"

DROP_COLS = ["device_fraud_count"]

# -1 이 결측 의미
NEG_ONE_AS_NA = [
    "prev_address_months_count",
    "bank_months_count",
    "current_address_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

# 음수 전체가 결측 의미
NEGATIVE_AS_NA = ["intended_balcon_amount"]

# 음수가 정상값이므로 절대 건드리지 않는 컬럼
KEEP_NEGATIVE = ["credit_risk_score", "velocity_6h"]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

TRAIN_MONTHS = [0, 1, 2, 3, 4, 5]
VALID_MONTHS = [6]
HOLDOUT_MONTHS = [7]  # 7주차 드리프트 실험 전까지 사용 금지

# 원본 검증용 기대값
EXPECTED_RAW_ROWS = 1_000_000
EXPECTED_RAW_COLS = 32
EXPECTED_SPLIT = {
    "train": {"rows": 794_989, "fraud": 8_151},
    "valid": {"rows": 108_168, "fraud": 1_450},
}


def load_raw(path: Path) -> pd.DataFrame:
    """원본 CSV 로드 후 형태 검증"""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. Kaggle에서 Base.csv를 받아 data/raw/ 에 두세요."
        )
    df = pd.read_csv(path)

    if df.shape != (EXPECTED_RAW_ROWS, EXPECTED_RAW_COLS):
        print(
            f"[warn] 원본 shape이 예상과 다릅니다: {df.shape} "
            f"(예상 {(EXPECTED_RAW_ROWS, EXPECTED_RAW_COLS)})",
            file=sys.stderr,
        )

    missing = [c for c in [TARGET, MONTH_COL] + CATEGORICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """정보량 없는 컬럼 드롭"""
    present = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=present)


def apply_missing_rules(df: pd.DataFrame) -> pd.DataFrame:
    """음수 결측 인코딩을 NaN으로 변환"""
    df = df.copy()

    for col in NEG_ONE_AS_NA:
        if col not in df.columns:
            continue
        df[col] = df[col].mask(df[col] == -1)

    for col in NEGATIVE_AS_NA:
        if col not in df.columns:
            continue
        df[col] = df[col].mask(df[col] < 0)

    return df


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """범주형 컬럼을 category dtype으로 변환

    반드시 month 분할 이전에 호출할 것.
    분할 후 각각 캐스팅하면 split마다 카테고리 집합이 달라져 LightGBM 학습/추론 시 코드가 어긋남.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def split_by_month(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """month 기준 시간 분할. month 7은 반환하지 않음"""
    train = df[df[MONTH_COL].isin(TRAIN_MONTHS)].reset_index(drop=True)
    valid = df[df[MONTH_COL].isin(VALID_MONTHS)].reset_index(drop=True)

    overlap = set(train.index) & set(valid.index)
    assert not (set(TRAIN_MONTHS) & set(VALID_MONTHS)), "train/valid month 구간이 겹칩니다"
    assert MONTH_COL in train.columns

    n_holdout = int(df[MONTH_COL].isin(HOLDOUT_MONTHS).sum())
    if n_holdout:
        print(f"[info] month 7 {n_holdout:,}행은 제외했습니다 (드리프트 실험 전용)")

    return {"train": train, "valid": valid}


def report(name: str, df: pd.DataFrame) -> None:
    """분할 결과 요약 출력 및 기대값 대조"""
    n = len(df)
    n_fraud = int(df[TARGET].sum())
    rate = n_fraud / n * 100 if n else 0.0
    print(f"  {name:<6} {n:>8,}행 / 사기 {n_fraud:>6,}건 / {rate:.3f}%")

    exp = EXPECTED_SPLIT.get(name)
    if exp and (n != exp["rows"] or n_fraud != exp["fraud"]):
        print(
            f"    [warn] 기대값과 다릅니다 "
            f"(예상 {exp['rows']:,}행 / 사기 {exp['fraud']:,}건)",
            file=sys.stderr,
        )


def save(df: pd.DataFrame, path: Path) -> None:
    """parquet 저장. category dtype 보존을 위해 pyarrow 엔진 사용"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    print(f"  saved -> {path}")


def run(raw_path: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    print(f"[1/5] load  {raw_path}")
    df = load_raw(raw_path)

    print("[2/5] drop columns")
    df = drop_columns(df)

    print("[3/5] missing rules")
    df = apply_missing_rules(df)

    print("[4/5] cast categoricals")
    df = cast_categoricals(df)

    print("[5/5] split by month")
    splits = split_by_month(df)
    for name, part in splits.items():
        report(name, part)
        save(part, out_dir / f"{name}.parquet")

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="BAF Base 전처리")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    run(args.raw, args.out)


if __name__ == "__main__":
    main()
