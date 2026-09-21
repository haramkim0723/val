"""Confusion Matrix (혼동행렬)

행 = 실제(정답) 클래스, 열 = 예측 클래스.
matrix[i][j] = 실제가 labels[i]인데 labels[j]로 예측한 개수.

이진 분류(labels=[0, 1])라면:
    [[TN, FP],
     [FN, TP]]
"""

from common import get_labels, validate_inputs


def confusion_matrix(y_true, y_pred, labels=None):
    """N x N 혼동행렬(list of list)을 반환. labels로 행/열 순서를 지정할 수 있다."""
    validate_inputs(y_true, y_pred)
    if labels is None:
        labels = get_labels(y_true, y_pred)

    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return matrix


def format_confusion_matrix(matrix, labels):
    """혼동행렬을 보기 좋은 문자열 표로 만든다."""
    names = [str(label) for label in labels]
    corner = "true\\pred"
    width = max(len(s) for s in names + [str(v) for row in matrix for v in row])

    lines = [corner + "".join(f" {name:>{width}}" for name in names)]
    for name, row in zip(names, matrix):
        lines.append(f"{name:>{len(corner)}}" + "".join(f" {v:>{width}}" for v in row))
    return "\n".join(lines)
