# LAS 필터링 결과 검증 (P / R / A / F1)

포인트클라우드(LAS)를 필터링한 결과(예측)를 기준 결과(정답)와 비교해서
**Precision, Recall, Accuracy, F1**과 혼동행렬을 계산한다.

## 사용법

필요 환경: Python 3.10+, numpy

```bash
python compare_las.py 정답.las 예측.las
```

| 인자 / 옵션 | 설명 |
|---|---|
| `정답.las` (첫 번째) | 기준이 되는 결과 |
| `예측.las` (두 번째) | 평가할 필터링 결과 |
| `--original 원본.las` | 필터링 전 원본 (선택). 주면 Accuracy가 정확해진다 |
| `--positive kept\|removed` | 양성 기준. `kept`(기본) = 남긴 점, `removed` = 제거한 점 (`--original` 필요) |
| `--output 파일` | 결과 저장. `.csv`는 표 한 줄, 그 외(`.txt`)는 화면과 같은 텍스트 |

```bash
# 정답이 원본에서 일부를 지운 파일일 때 (권장)
python compare_las.py data\정답.las data\예측.las --original data\원본.las --output 결과.csv
```

출력 예:

```
true\pred     kept  removed
     kept 40826397   465695
  removed        0        0

TP=40,826,397  FP=0  FN=465,695  TN=0
Accuracy : 0.9887
Precision: 1.0000
Recall   : 0.9887
F1       : 0.9943
```

## 작동 원리

**1. 문제 정의.** 각 점을 "남김(kept) / 제거(removed)"로 분류한 문제로 본다.
정답 파일에 남아 있으면 정답=kept, 예측 파일에 남아 있으면 예측=kept.
양성(positive)은 기본적으로 kept다. Classification 값은 쓰지 않고, 점이 파일에 **있는지 없는지**만 본다.

**2. LAS 읽기 (`las_io.py`).** 헤더에서 점 개수, 레코드 길이, scale/offset을 읽고,
점 데이터는 `numpy.memmap`으로 열어 X, Y, Z(int32, 레코드의 0/4/8 바이트)만 4백만 점씩 끊어서 읽는다.
그래서 4천만 점(1.4GB)도 한 번에 메모리에 올리지 않는다.

**3. 점 매칭 (`compare_las.py`).** 파일마다 점 순서가 다를 수 있으므로 좌표로 매칭한다.
- 실제 좌표(`정수 × scale + offset`)를 기준 파일의 정수 격자로 변환하고 반올림한다.
  파일마다 LAS 버전, scale, offset이 달라도 같은 점이 같은 값이 된다.
- `(x, y, z)` 세 정수를 정확히 하나의 정수 키(`(x·Sy + y)·Sz + z`, S는 축별 범위)로 합친다.
  이후 비교는 정수 정렬과 이진 탐색으로 빠르게 한다.
- 전체 점 범위 밖에 있는 점은 계산에서 제외하고 리포트에 개수를 알려준다.

**4. 중복 좌표.** 좌표가 완전히 같은 점이 여러 개 있으면 키별 **개수**로 처리한다.
키마다 `TP = min(정답 개수, 예측 개수)`로 세고, 어느 점이 남았는지 알 수 없는 점은 리포트에 "모호한 점"으로 표시한다.
필터가 좌표를 보고 판단하면 같은 좌표의 점은 함께 남거나 제거되므로 모호함이 없다.

**5. 혼동행렬.**

| | 예측 kept | 예측 removed |
|---|---|---|
| **정답 kept** | TP | FN |
| **정답 removed** | FP | TN |

- 원본을 줄 때: 전체 점 = 원본. 둘 다 지운 점까지 TN으로 센다.
- 원본이 없을 때: 전체 점 = 정답 ∪ 예측. 둘 다 지운 점은 알 수 없어서 TN=0이고, Accuracy는 참고용이다.

**6. 지표.**

| 지표 | 식 | 의미 |
|---|---|---|
| Precision | TP / (TP + FP) | 남겼다고 한 것 중 실제로 남길 점의 비율 |
| Recall | TP / (TP + FN) | 남겨야 할 점 중 실제로 남긴 비율 |
| Accuracy | (TP + TN) / 전체 | 남김/제거 판단이 맞은 비율 |
| F1 | 2·P·R / (P + R) | Precision과 Recall의 조화평균 |

분모가 0이면 0으로 처리한다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `compare_las.py` | 실행 진입점. 점 매칭, 개수 계산, 리포트/저장 |
| `las_io.py` | LAS 헤더/좌표 읽기, 좌표 격자 변환 |
| `precision.py` `recall.py` `accuracy.py` `f1_score.py` | 지표별 계산 함수 |
| `confusion_matrix.py` | 혼동행렬 생성과 표 출력 |
| `common.py` | 입력 검증, 개수 세기, 0 나누기 처리 |

## 주의

- `.las`만 지원한다. `.laz`(압축)는 `.las`로 변환해서 사용한다.
- 세 파일은 같은 좌표계여야 한다. 좌표가 다르면 "원본에 없는 점"으로 잡힌다.
- 양성이 kept이고 제거된 점이 적으면(예: 1%) Accuracy가 높게 나온다. 제거가 목적이면 `--positive removed`를 함께 본다.
- 4천만 점 기준 실행 시간은 15초 안팎이다.
