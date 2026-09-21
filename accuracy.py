"""Accuracy (정확도) = 맞춘 개수 / 전체 개수"""

from common import validate_inputs, safe_divide


def accuracy_from_counts(tp, fp, fn, tn):
    """이미 센 TP, FP, FN, TN으로 accuracy 계산 (포인트클라우드처럼 큰 데이터용)."""
    return safe_divide(tp + tn, tp + fp + fn + tn)


def accuracy(y_true, y_pred):
    validate_inputs(y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return safe_divide(correct, len(y_true))
