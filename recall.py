from common import (
    confusion_counts,
    get_labels,
    safe_divide,
    validate_average,
    validate_inputs,
)


def recall_from_counts(tp, fn, zero_division=0.0):
    return safe_divide(tp, tp + fn, zero_division)


def class_recall(y_true, y_pred, label, zero_division=0.0):
    tp, _, fn, _ = confusion_counts(y_true, y_pred, label)
    return recall_from_counts(tp, fn, zero_division)


def recall(y_true, y_pred, average="binary", pos_label=1, zero_division=0.0):
    validate_inputs(y_true, y_pred)
    validate_average(average, y_true, y_pred)

    if average == "binary":
        return class_recall(y_true, y_pred, pos_label, zero_division)

    labels = get_labels(y_true, y_pred)
    if average == "macro":
        scores = [class_recall(y_true, y_pred, l, zero_division) for l in labels]
        return sum(scores) / len(scores)

    # micro
    tp_sum = fn_sum = 0
    for label in labels:
        tp, _, fn, _ = confusion_counts(y_true, y_pred, label)
        tp_sum += tp
        fn_sum += fn
    return safe_divide(tp_sum, tp_sum + fn_sum, zero_division)
