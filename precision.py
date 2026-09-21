"""Precision (정밀도) = TP / (TP + FP)

"양성이라고 예측한 것 중에 진짜 양성인 비율"
"""

from common import (
    confusion_counts,
    get_labels,
    safe_divide,
    validate_average,
    validate_inputs,
)


def precision_from_counts(tp, fp, zero_division=0.0):
    """이미 센 TP, FP로 precision 계산 (포인트클라우드처럼 큰 데이터용)."""
    return safe_divide(tp, tp + fp, zero_division)


def class_precision(y_true, y_pred, label, zero_division=0.0):
    """`label` 클래스 하나에 대한 precision (입력 검증 없음)."""
    tp, fp, _, _ = confusion_counts(y_true, y_pred, label)
    return precision_from_counts(tp, fp, zero_division)


def precision(y_true, y_pred, average="binary", pos_label=1, zero_division=0.0):
    """
    average:
        "binary": pos_label 클래스 하나에 대해서만 계산
        "macro" : 클래스별 precision의 단순 평균
        "micro" : 모든 클래스의 TP, FP를 합산한 뒤 계산
    """
    validate_inputs(y_true, y_pred)
    validate_average(average, y_true, y_pred)

    if average == "binary":
        return class_precision(y_true, y_pred, pos_label, zero_division)

    labels = get_labels(y_true, y_pred)
    if average == "macro":
        scores = [class_precision(y_true, y_pred, l, zero_division) for l in labels]
        return sum(scores) / len(scores)

    # micro
    tp_sum = fp_sum = 0
    for label in labels:
        tp, fp, _, _ = confusion_counts(y_true, y_pred, label)
        tp_sum += tp
        fp_sum += fp
    return safe_divide(tp_sum, tp_sum + fp_sum, zero_division)
