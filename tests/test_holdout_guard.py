"""홀드아웃 차단 가드.

split_by_month의 기본 동작이 차단(deny)임을 지킨다.

이 테스트가 깨졌다면 홀드아웃이 기본으로 열렸다는 뜻이고, 그 상태로 학습이
한 번이라도 돌면 7주차 드리프트 실험이 무의미해진다. 오염은 되돌릴 수 없고
(month 8이 없다) 증상도 3단계에 가서야 간접적으로 나타나므로,
사람의 확인이 아니라 CI가 차단한다.

원본 데이터를 쓰지 않는다. month 컬럼과 target만 있으면 되므로 CI에서 돈다.
"""

from __future__ import annotations

import pandas as pd

from preprocess import (
    HOLDOUT_MONTHS,
    MONTH_COL,
    TARGET,
    TRAIN_MONTHS,
    VALID_MONTHS,
    split_by_month,
)

ROWS_PER_MONTH = 3


def _frame() -> pd.DataFrame:
    months = TRAIN_MONTHS + VALID_MONTHS + HOLDOUT_MONTHS
    return pd.DataFrame(
        {
            MONTH_COL: [m for m in months for _ in range(ROWS_PER_MONTH)],
            TARGET: 0,
        },
        index=range(len(months) * ROWS_PER_MONTH),
    )


def test_holdout_denied_by_default():
    assert set(split_by_month(_frame())) == {"train", "valid"}


def test_holdout_rows_do_not_leak_into_other_splits():
    # 키가 없는 것만으로는 부족하다. 홀드아웃 행이 train/valid에 섞이지 않아야 한다.
    for name, part in split_by_month(_frame()).items():
        assert not part[MONTH_COL].isin(HOLDOUT_MONTHS).any(), f"{name}에 홀드아웃 유입"


def test_holdout_opens_only_with_explicit_flag():
    splits = split_by_month(_frame(), include_holdout=True)

    assert set(splits) == {"train", "valid", "holdout"}
    assert splits["holdout"][MONTH_COL].isin(HOLDOUT_MONTHS).all()
    assert len(splits["holdout"]) == len(HOLDOUT_MONTHS) * ROWS_PER_MONTH
