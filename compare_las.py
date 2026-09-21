"""예측 LAS를 정답 LAS와 비교해서 P / R / A / F1을 계산한다.

각 점을 "남김(kept) / 제거(removed)"로 분류한 문제로 본다.
    정답 = 첫 번째 LAS에 남아 있는가
    예측 = 두 번째 LAS에 남아 있는가
점은 XYZ 좌표로 매칭하므로 점 순서가 달라도 된다.
좌표가 완전히 같은 중복 점은 개수로 처리한다 (남은 개수의 min을 일치로 본다).

원본(--original)을 주면 두 파일에 모두 없는 점(TN)까지 세서 Accuracy가 정확해진다.
원본이 없으면 정답과 예측의 합집합을 전체 점으로 보고 TN=0으로 계산한다.

사용법:
    python compare_las.py 정답.las 예측.las [--original 원본.las] [--positive kept|removed] [--output 결과.csv]
"""

import argparse
import csv

import numpy as np

from accuracy import accuracy_from_counts
from confusion_matrix import format_confusion_matrix
from f1_score import f1_from_counts
from las_io import iter_xyz_chunks, read_header, to_ref_grid
from precision import precision_from_counts
from recall import recall_from_counts

OUTSIDE = -1  # 전체 점의 좌표 범위 밖에 있는 점에 붙이는 키


def axis_ranges(ref, infos):
    """`infos` 파일들의 좌표를 ref 정수 격자로 봤을 때 축별 최솟값과 격자 크기(span)."""
    lo = [None] * 3
    hi = [None] * 3
    for info in infos:
        for chunk in iter_xyz_chunks(info):
            for axis, raw in enumerate(chunk):
                values = to_ref_grid(raw, axis, info, ref)
                a, b = int(values.min()), int(values.max())
                lo[axis] = a if lo[axis] is None else min(lo[axis], a)
                hi[axis] = b if hi[axis] is None else max(hi[axis], b)
    if lo[0] is None:
        raise ValueError("점이 하나도 없는 LAS 파일입니다.")
    spans = [h - l + 1 for l, h in zip(lo, hi)]
    if spans[0] * spans[1] * spans[2] >= 2**63:
        raise ValueError("좌표 범위가 너무 커서 점을 하나의 정수 키로 만들 수 없습니다.")
    return lo, spans


def point_keys(info, ref, lo, spans):
    """각 점의 (x, y, z)를 ref 격자 기준의 정수 키 하나로 만든다. 범위 밖이면 OUTSIDE."""
    keys = np.empty(info.num_points, dtype=np.int64)
    pos = 0
    for chunk in iter_xyz_chunks(info):
        g = [to_ref_grid(v, axis, info, ref) - lo[axis] for axis, v in enumerate(chunk)]
        inside = np.ones(len(g[0]), dtype=bool)
        for axis in range(3):
            inside &= (g[axis] >= 0) & (g[axis] < spans[axis])
        key = (g[0] * spans[1] + g[1]) * spans[2] + g[2]
        keys[pos : pos + len(key)] = np.where(inside, key, OUTSIDE)
        pos += len(key)
    return keys


def unique_counts(keys):
    """(정렬된 고유 키, 개수, 범위 밖 점 개수). 범위 밖(OUTSIDE)은 고유 키에서 뺀다."""
    unique, counts = np.unique(keys, return_counts=True)
    outside = 0
    if len(unique) and unique[0] == OUTSIDE:
        outside = int(counts[0])
        unique, counts = unique[1:], counts[1:]
    return unique, counts.astype(np.int32), outside


def compare_kept_points(truth_path, pred_path, original_path=None):
    """TP/FP/FN/TN 개수 (양성 = 남김)와 진단 정보를 반환.

    original_path가 없으면 정답 ∪ 예측을 전체 점으로 본다 (TN은 항상 0).
    """
    truth_info = read_header(truth_path)
    pred_info = read_header(pred_path)
    original_info = read_header(original_path) if original_path else None

    ref = original_info or truth_info
    universe_files = [original_info] if original_info else [truth_info, pred_info]
    lo, spans = axis_ranges(ref, universe_files)

    truth_keys, truth_counts, truth_outside = unique_counts(point_keys(truth_info, ref, lo, spans))
    pred_keys, pred_counts, pred_outside = unique_counts(point_keys(pred_info, ref, lo, spans))

    if original_info:
        univ_keys, univ_counts, _ = unique_counts(point_keys(original_info, ref, lo, spans))
    else:
        # 합집합: 키별 개수는 두 파일 중 더 많은 쪽
        univ_keys = np.union1d(truth_keys, pred_keys)
        univ_counts = np.zeros(len(univ_keys), dtype=np.int32)
        for keys, counts in ((truth_keys, truth_counts), (pred_keys, pred_counts)):
            idx = np.searchsorted(univ_keys, keys)
            univ_counts[idx] = np.maximum(univ_counts[idx], counts)
    total = int(univ_counts.sum())

    def kept_per_key(keys, counts, outside):
        """전체 점의 고유 키별로 해당 파일에 남아 있는 점 개수와, 전체에 없는 점 개수."""
        idx = np.minimum(np.searchsorted(univ_keys, keys), len(univ_keys) - 1)
        found = univ_keys[idx] == keys
        kept = np.zeros(len(univ_keys), dtype=np.int32)
        kept[idx[found]] = np.minimum(counts[found], univ_counts[idx[found]])
        extra = outside + int(counts[~found].sum())
        extra += int(np.maximum(counts[found] - univ_counts[idx[found]], 0).sum())
        return kept, extra

    truth_kept, truth_extra = kept_per_key(truth_keys, truth_counts, truth_outside)
    pred_kept, pred_extra = kept_per_key(pred_keys, pred_counts, pred_outside)

    tp = int(np.minimum(pred_kept, truth_kept).sum())
    fp = int(pred_kept.sum()) - tp
    fn = int(truth_kept.sum()) - tp
    tn = total - tp - fp - fn

    # 중복 좌표 점 중, 일부만 남아서 어느 점이 남았는지 알 수 없는 점 수
    partial = (pred_kept > 0) & (pred_kept < univ_counts)
    partial |= (truth_kept > 0) & (truth_kept < univ_counts)
    ambiguous = int(univ_counts[partial & (univ_counts > 1)].sum())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "has_original": original_info is not None,
        "pred_extra": pred_extra,
        "truth_extra": truth_extra,
        "ambiguous": ambiguous,
    }


def swap_positive(tp, fp, fn, tn):
    """양성을 '남김'에서 '제거'로 바꾼 (TP, FP, FN, TN)."""
    return tn, fn, fp, tp


def compute_metrics(result, positive="kept"):
    """양성 기준에 맞춰 TP/FP/FN/TN과 A/P/R/F1을 계산."""
    tp, fp, fn, tn = (result[k] for k in ("tp", "fp", "fn", "tn"))
    if positive == "removed":
        tp, fp, fn, tn = swap_positive(tp, fp, fn, tn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy_from_counts(tp, fp, fn, tn),
        "precision": precision_from_counts(tp, fp),
        "recall": recall_from_counts(tp, fn),
        "f1": f1_from_counts(tp, fp, fn),
    }


def format_report(result, positive="kept"):
    m = compute_metrics(result, positive)
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]

    labels = ["kept", "removed"] if positive == "kept" else ["removed", "kept"]
    matrix = [[tp, fn], [fp, tn]]
    total_label = "원본 점 개수" if result["has_original"] else "전체 점 개수(정답 ∪ 예측)"

    lines = [
        f"{total_label}: {result['total']:,}",
        f"양성(positive): {positive}",
        "",
        "혼동행렬 (행=정답, 열=예측)",
        format_confusion_matrix(matrix, labels),
        "",
        f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}",
        f"Accuracy : {m['accuracy']:.4f}",
        f"Precision: {m['precision']:.4f}",
        f"Recall   : {m['recall']:.4f}",
        f"F1       : {m['f1']:.4f}",
    ]

    notes = []
    if not result["has_original"]:
        notes.append(
            "원본(--original)이 없어서 두 파일에 모두 없는 점(TN)을 알 수 없습니다 (TN=0으로 계산).\n"
            "  Accuracy는 TP / (TP + FP + FN)이라 참고용입니다."
        )
    if result["pred_extra"] or result["truth_extra"]:
        notes.append(
            f"원본에 없는 점: 예측 {result['pred_extra']:,}개, "
            f"정답 {result['truth_extra']:,}개 (계산에서 제외됨)"
        )
    if result["ambiguous"]:
        notes.append(
            f"중복 좌표 때문에 어느 점이 남았는지 모호한 점: {result['ambiguous']:,}개"
        )
    if notes:
        lines += ["", "[참고]"] + notes
    return "\n".join(lines)


def write_output(path, args, result):
    """.csv면 표 한 줄, 그 외에는 화면과 같은 텍스트로 저장 (엑셀/메모장에서 한글이 깨지지 않게 utf-8-sig)."""
    if path.lower().endswith(".csv"):
        m = compute_metrics(result, args.positive)
        row = {
            "truth": args.truth,
            "pred": args.pred,
            "original": args.original or "",
            "positive": args.positive,
            "total": result["total"],
            **m,
        }
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    else:
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write(format_report(result, args.positive) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("truth", help="정답 LAS (첫 번째, 기준)")
    parser.add_argument("pred", help="예측 LAS (두 번째, 평가할 결과)")
    parser.add_argument("--original", metavar="원본.las",
                        help="필터링 전 원본 LAS (선택). 주면 TN까지 세서 Accuracy가 정확해짐")
    parser.add_argument("--positive", choices=("kept", "removed"), default="kept",
                        help="양성 클래스: 남긴 점(kept, 기본) 또는 제거한 점(removed, --original 필요)")
    parser.add_argument("--output", metavar="파일",
                        help="결과 저장 경로. .csv면 표 한 줄, 그 외(.txt 등)는 화면과 같은 텍스트")
    args = parser.parse_args()

    if args.positive == "removed" and not args.original:
        parser.error("--positive removed는 제거된 점을 알아야 하므로 --original이 필요합니다.")

    result = compare_kept_points(args.truth, args.pred, args.original)
    print(format_report(result, args.positive))
    if args.output:
        write_output(args.output, args, result)
        print(f"\n저장됨: {args.output}")


if __name__ == "__main__":
    main()
