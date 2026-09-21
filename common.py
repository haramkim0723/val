VALID_AVERAGES = ("binary", "macro", "micro")


def validate_inputs(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true와 y_pred의 길이가 다릅니다: {len(y_true)} != {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("입력이 비어 있습니다.")


def get_labels(y_true, y_pred):
    return sorted(set(y_true) | set(y_pred), key=str)


def validate_average(average, y_true, y_pred):
    if average not in VALID_AVERAGES:
        raise ValueError(f"average는 {VALID_AVERAGES} 중 하나여야 합니다: {average!r}")
    if average == "binary" and len(get_labels(y_true, y_pred)) > 2:
        raise ValueError(
            "클래스가 3개 이상입니다. average='macro' 또는 'micro'를 사용하세요."
        )


def confusion_counts(y_true, y_pred, label):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p == label:
            if t == label:
                tp += 1
            else:
                fp += 1
        else:
            if t == label:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def safe_divide(numerator, denominator, zero_division=0.0):
    if denominator == 0:
        return zero_division
    return numerator / denominator
