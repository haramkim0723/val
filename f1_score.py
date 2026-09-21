"""F1 score = 2 * P * R / (P + R)

precision과 recall의 조화평균.
"""

from common import (
    confusion_counts,
    get_labels,
    safe_divide,
    validate_average,
    validate_inputs,
)
from precision import precision, precision_from_counts
from recall import recall, recall_from_counts


def _harmonic_mean(p, r):
    # P + R이 0이면 F1도 0 (zero_division은 P, R을 구할 때 이미 적용됨)
    return safe_divide(2 * p * r, p + r)


def f1_from_counts(tp, fp, fn, zero_division=0.0):
    """이미 센 TP, FP, FN으로 F1 계산 (포인트클라우드처럼 큰 데이터용)."""
    p = precision_from_counts(tp, fp, zero_division)
    r = recall_from_counts(tp, fn, zero_division)
    return _harmonic_mean(p, r)


def class_f1(y_true, y_pred, label, zero_division=0.0):
    """`label` 클래스 하나에 대한 F1 (입력 검증 없음)."""
    tp, fp, fn, _ = confusion_counts(y_true, y_pred, label)
    return f1_from_counts(tp, fp, fn, zero_division)


def f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0.0):
    """
    average:
        "binary": pos_label 클래스 하나에 대해서만 계산
        "macro" : 클래스별 F1의 단순 평균
        "micro" : micro precision과 micro recall로 계산
    """
    validate_inputs(y_true, y_pred)
    validate_average(average, y_true, y_pred)

    if average == "binary":
        return class_f1(y_true, y_pred, pos_label, zero_division)

    if average == "macro":
        scores = [
            class_f1(y_true, y_pred, l, zero_division)
            for l in get_labels(y_true, y_pred)
        ]
        return sum(scores) / len(scores)

    # micro
    p = precision(y_true, y_pred, "micro", zero_division=zero_division)
    r = recall(y_true, y_pred, "micro", zero_division=zero_division)
    return _harmonic_mean(p, r)
