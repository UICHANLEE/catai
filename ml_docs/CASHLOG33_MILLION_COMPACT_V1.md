# CashLog33 100만 뷰 경량 모델 설계·실행 기록

## 1. 목적

- CashLog 33개 leaf 계약은 그대로 유지한다.
- 이미지 단독으로 구분 가능한 23개 leaf는 경량 시각 모델이 담당한다.
- 주거·금융·통신요금·미분류처럼 시각 객체만으로 의미를 확정할 수 없는
  10개 leaf는 기존 RapidOCR, 한국어 텍스트 모델, OCR lexicon, 안전 fallback이
  담당한다.
- 원본 이미지 수와 증강으로 만든 논리 학습 뷰 수를 절대 같은 의미로 쓰지 않는다.
- 후보가 현재 배포 모델보다 좋아진 경우에만 INT8 ONNX 모델로 교체한다.

## 2. 모델 구조

```text
입력 이미지
  ├─ MobileNetV4 Conv Small → 23개 시각 leaf logits
  └─ RapidOCR → 한국어 텍스트
                    ├─ TF-IDF/SGD 33개 leaf 확률
                    └─ 감사 가능한 OCR lexicon 점수
                                  ↓
                      33개 leaf 확률 융합
                                  ↓
                  Top-3 + 사용자 확인 필요 여부
```

- backbone: `mobilenetv4_conv_small.e2400_r224_in1k`
- pretrained source: `timm/mobilenetv4_conv_small.e2400_r224_in1k`
- pretrained license: Apache-2.0
- pretrained SHA-256:
  `7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4`
- 사전학습 입력 계약: bicubic resize 256, center crop 224, ImageNet mean/std
- 분류기 출력: 학습 가능한 23개 시각 leaf
- 학습 최종 가중치: EMA
- 운영 시각 모델: ONNX Runtime 정적 INT8 QDQ
- 기존 SigLIP 경로는 설정 파일로 롤백 가능하게 유지하되 compact 후보에서는 로드하지 않는다.

## 3. 원본 데이터

### 3.1 신규 Open Images 원본

- 데이터셋: Open Images V7 train human-verified image-level labels
- class 설명:
  `https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv`
- 사람 검수 train label:
  `https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv`
- 이미지 귀속·라이선스 metadata:
  `https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv`
- 이미지 object API:
  `https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg`
- mapped unique 후보: 505,603
- 여러 CashLog leaf에 동시에 매핑되어 제거: 36,597
- 모호성 제거 후 후보: 469,006
- metadata에서 CC BY 2.0을 확인한 후보: 469,006
- 클래스당 상한: 없음
- 실제 다운로드·디코딩·크기 검증 통과: 469,001
- 영구 제외: 5개. 모두 원본 응답이 12MB를 초과했다.
- 일시적인 S3 연결 실패는 최대 6회 재시도하는 2차 수집에서 모두 복구했다.
- 최종 manifest SHA-256:
  `f3a1f2ac377357aea853e5b2f3d0520509b2930f7ddda92a89a64988123dccce`
- 최종 수집 기준은
  `data/raw/cashlog33/openimages_v7_train/collection_summary.json`이다.

Open Images annotation은 CC BY 4.0이며, 각 이미지 row는 metadata의
`License == https://creativecommons.org/licenses/by/2.0/` 조건을 통과해야 한다.
manifest에는 원본 landing URL, 저작자, 제목, 라이선스 URL, attribution을 보존한다.

### 3.2 기존 원본

- `originals_v2_additional`: 24,013 rows
- Amazon Berkeley Objects: 23,272
- Open Products 계열 API 및 Openverse: 739
- 사용자 동의 actual 사진: 2
- 기존 원본의 상세 출처와 라이선스 감사:
  `reports/cashlog33/originals_v2/source_audit.json`

Open Images validation 17,983장은 학습 원본에 합치지 않는다. 이 중 기존에 고정한
3,596장 test split만 외부 시각 proxy 비교에 사용한다.

## 4. 정제와 split

1. Open Images의 positive confidence가 1인 human-verified label만 사용한다.
2. 한 이미지가 둘 이상의 CashLog leaf에 걸리면 제거한다.
3. 디코딩 실패, 12MB 초과, 4천만 pixel 초과, 최소 변 96 미만 이미지는 제거한다.
4. 저장 JPEG의 SHA-256으로 출처 간 exact duplicate를 제거한다.
5. 동일 SHA-256이 서로 다른 leaf에 있으면 전부 제거한다.
6. 공식 Open Images validation 외부 test와 SHA-256 또는 `source_id`가 겹치면
   빌드를 실패시킨다.
7. Open Images train 원본만 leaf별 deterministic 80/20 train/validation split을 한다.
8. 기존 API·ABO·actual 원본은 이전 검증셋으로 유출하지 않고 train에만 둔다.

## 5. 100만 뷰 스케줄

- `1,000,000 views`는 100만 개의 새 원본이라는 뜻이 아니다.
- 한 view는 `원본 row index + augmentation seed` 한 쌍이다.
- 학습 가능한 모든 고유 train 원본을 최소 1회 반드시 스케줄에 넣는다.
- 남는 view는 원본 수가 적은 leaf부터 water-filling으로 보충한다.
- 원본이 많은 의류도 버리지 않으며, 스케줄 역빈도 class weight로 loss 기여도를 맞춘다.
- 정확한 row index와 seed는
  `data/processed/cashlog33/training/million_v1/million_view_schedule.npz`에 저장한다.

## 6. 증강

- RandomResizedCrop: scale 0.65~1.00, ratio 0.75~1.3333, bicubic
- RandomHorizontalFlip: p=0.5
- ColorJitter: brightness 0.20, contrast 0.20, saturation 0.15, hue 0.02
- RandAugment: 2 operations, magnitude 7
- RandomErasing: p=0.10, scale 0.02~0.10
- 총 4 epochs, epoch당 250,000 logical views
- 모든 augmentation seed를 스케줄에 기록해 재현 가능하게 한다.

검증에는 증강을 쓰지 않고 bicubic resize 256 → center crop 224만 사용한다.

## 7. 학습 설정

- device: Apple MPS
- batch size: 128
- optimizer: AdamW
- backbone LR: 1e-4
- classifier LR: 5e-4
- weight decay: 1e-4
- label smoothing: 0.05
- class weighting: 실제 100만 스케줄 역빈도, 평균 1로 정규화
- scheduler: cosine annealing
- EMA decay: 0.9997, warmup 사용
- best 선택 기준: 내부 validation Macro-F1
- probability calibration: 내부 validation logits로 scalar temperature 탐색
- calibration 기록: 보정 전·후 NLL과 15-bin ECE
- MPS 실측 steady-state: 약 283 images/s

MLflow experiment는 `cashlog33-million-compact`, run name은
`mobilenetv4-million-v1`이다. epoch별 loss, Top-1, Top-3, Macro-F1,
누적 view, 데이터 요약, 증강 계획, 체크포인트를 기록한다. 한국어 실행 로그는
`checkpoints/cashlog33/mobilenet_million_v1/run_log_ko.jsonl`에 기록한다.
학습 run ID는 `mlflow_run.json`에 보존하며, ONNX export 단계가 같은 run을 다시
열어 FP32, INT8, labels, export report를 `onnx/` artifact로 업로드한다.

## 8. ONNX와 양자화

1. best checkpoint의 EMA 가중치를 불러온다.
2. dynamic batch, input `images`, output `logits`, opset 18로 FP32 ONNX를 만든다.
3. ONNX Runtime graph optimization, symbolic/ONNX shape inference를 실행한다.
4. 내부 validation에서 23개 leaf class-balanced round-robin으로 512장을 뽑는다.
5. MinMax calibration, QDQ, activation QUInt8, weight QInt8, per-channel로 정적 양자화한다.
6. PyTorch↔FP32 ONNX 오차, FP32↔INT8 예측 일치율, 정확도 차이, SHA-256,
   파일 크기, CPU 단일 이미지 p50/p95를 기록한다.
7. 게이트에 사용할 INT8 ONNX, labels, export report를
   `src/catai/assets/cashlog33_mobilenetv4_int8_v1/`에 checksum 검증 후
   원자적으로 복사한다. 이 경량 배포 자산만 Git과 Python package에 포함한다.

내보내기 스모크 테스트에서는 같은 구조의 미학습 모델이 FP32 약 9.6MB,
INT8 약 2.7MB였으며 동적 배치 ONNX Runtime 추론을 통과했다. 최종 크기는
학습 후 `export_report.json`을 기준으로 한다.

## 9. ONNX I/O

```text
input
  name: images
  dtype: float32
  shape: [N, 3, 224, 224]
  normalization:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

output
  name: logits
  dtype: float32
  shape: [N, 23]
  labels: onnx/labels.json 순서
```

서빙기는 logits에 softmax를 적용하고 23개 점수를 33개 taxonomy 공간에 배치한다.
softmax 전에는 best checkpoint의 내부 validation에서 고정한 scalar temperature로
logits를 나눠 확률을 보정한다.
나머지 10개 leaf는 OCR·텍스트·lexicon 경로가 담당한다. 외부 API 응답 계약은
기존 `recommended_category`, `top_categories`, `need_user_check`를 유지한다.

## 10. 비교와 승격 기준

고정 외부 proxy는 Open Images validation의 기존 test 3,596장이다. 이 데이터는
신규 모델의 공식 train split과 분리되어 있다. 실제 배포 시각 경로를 다시 실행한
baseline은 다음과 같다.

| 지표 | 현재 배포 baseline |
|---|---:|
| Top-1 | 68.69% |
| Top-3 | 90.66% |
| Macro-F1 | 51.94% |
| 시각 추론 p50 | 54.15ms |
| 시각 stack 크기 | 약 1,497MB |

승격은 아래 조건을 모두 만족해야 한다.

- INT8 Top-1이 현재 배포 baseline보다 높다.
- INT8 Macro-F1이 현재 배포 baseline보다 높다.
- 표본 30장 이상 leaf의 recall 최대 하락이 10%p 이내다.
- INT8 Top-1 하락이 FP32 대비 0.5%p 이내다.
- INT8 모델이 현재 시각 stack보다 작다.
- INT8 단일 이미지 p50이 현재 시각 stack보다 빠르다.

실제 CashLog 사진은 현재 2장뿐이고 사용자 피드백 학습 원본에도 포함되어 있다.
따라서 독립 holdout이나 승격 gate로 사용하지 않는다. 기존·후보의 재생 결과와
학습 SHA-256 중복 수는 smoke 진단으로 기록하되, 95% 정확도나 모델 개선의
근거로 사용하지 않는다. 새 실제 사진은 앞으로 수집 시점에 train과 고정
actual holdout으로 먼저 분리해야 한다.

## 11. 승격과 롤백

- 후보 config:
  `configs/cashlog/hybrid.million-int8-candidate.json`
- 버전 관리 배포 자산:
  `src/catai/assets/cashlog33_mobilenetv4_int8_v1/`
- 비교 report:
  `reports/cashlog33/million_v1/compact_comparison.json`
- gate 통과 시 기존 serving config를
  `configs/cashlog/hybrid.serving.previous.json`으로 백업한다.
- 평가 당시 serving config SHA-256과 평가된 INT8 artifact SHA-256이 교체
  직전 파일과 모두 일치해야 한다.
- 그다음 후보를 `configs/cashlog/hybrid.serving.json`에 원자적으로 교체한다.
- 후보에서는 중복 식사 specialist를 비활성화하고 INT8 ONNX 시각 경로만 사용한다.
- API 재시작, health, 실제 사진 2장, 외부 요청 E2E를 확인한다.
- `/health`의 `vision_backend=compact_onnx`와
  `vision_providers=["CPUExecutionProvider"]`를 확인한다.
- 이상이 있으면 previous config로 즉시 복구한다.

## 12. 한계

- Open Images 객체 label은 구매 의도 자체를 증명하지 않는 visual proxy다.
- SHA-256 exact duplicate와 공식 source ID 누수는 차단하지만, 서로 다른 crop이나
  재인코딩으로 생긴 perceptual near-duplicate까지 완전 제거한 것은 아니다.
- 33개 leaf 중 10개는 이미지 단독 정답이 정의되기 어려워 OCR·텍스트가 필수다.
- `gift_event`, `health_gym`, `health_med` 등 일부 leaf의 Open Images 원본은 적다.
- 실제 CashLog 사진이 2장뿐이므로 95% 실서비스 정확도를 주장할 수 없다.
- 실제 사용자 수정 데이터가 쌓이면 고정 actual holdout과 재학습 train을 분리해야 한다.

## 13. 실행 산출물

- 수집 요약: `data/raw/cashlog33/openimages_v7_train/collection_summary.json`
- 데이터 요약: `data/processed/cashlog33/training/million_v1/dataset_summary.json`
- 증강 계획: `data/processed/cashlog33/training/million_v1/augmentation_plan.json`
- 학습 지표: `checkpoints/cashlog33/mobilenet_million_v1/metrics.csv`
- 한국어 로그: `checkpoints/cashlog33/mobilenet_million_v1/run_log_ko.jsonl`
- FP32/INT8: `checkpoints/cashlog33/mobilenet_million_v1/onnx/`
- 최종 배포 INT8·labels·provenance:
  `src/catai/assets/cashlog33_mobilenetv4_int8_v1/`
- 양자화 report: `checkpoints/cashlog33/mobilenet_million_v1/onnx/export_report.json`
- 최종 비교: `reports/cashlog33/million_v1/compact_comparison.json`

## 14. 실제 실행 결과 (2026-07-30)

### 14.1 데이터 수량과 무결성

| 항목 | 수량 |
|---|---:|
| 신규 Open Images 원본 | 469,001 |
| 기존 원본 | 24,013 |
| 전체 고유 원본 | 493,014 |
| train 원본 | 399,213 |
| validation 원본 | 93,801 |
| 23개 시각 leaf 학습 가능 train 원본 | 399,198 |
| 논리 학습 뷰 | 1,000,000 |
| 평균 원본 재사용 계수 | 2.5049 |

- 23개 학습 가능 leaf의 모든 원본 399,198장을 스케줄에 최소 1회 넣었다.
- 원본 수가 적은 leaf는 water-filling으로 보충했고, 원본이 가장 많은
  `fashion_clothes` 237,739장은 줄이지 않았다.
- 저장 JPEG SHA-256 duplicate: 0건
- cross-leaf SHA-256 conflict: 0건
- 외부 고정 test SHA-256 overlap: 0건
- Open Images source ID overlap: 0건
- 학습 스케줄 SHA-256:
  `7e931895bd71de522f646bd4e798981342dcc00f4fda0af6f249affa5deddc4b`

여기서 1,000,000은 새 원본 100만 장이 아니라, 위 고유 원본을 증강해 실제로
모델에 입력한 정확한 뷰 수다. 원본 수와 증강 뷰 수를 합치거나 서로 바꿔
표현하지 않는다.

### 14.2 MPS 학습

- MLflow experiment: `cashlog33-million-compact`
- MLflow run ID: `cf9524b498104f02b9749d7493357466`
- 장치: Apple MPS
- 총 학습 시간: 약 1시간 55분
- 실제 처리한 학습 뷰: 1,000,000
- best checkpoint: 1 epoch, 누적 250,000 views
- best checkpoint SHA-256:
  `4c81ca79a65c64bd5ea9d259e55b094762de4691cf8af28f3bd1a871e4a0b3f5`

| epoch | 누적 views | train Top-1 | validation Top-1 | Top-3 | Macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 250,000 | 60.46% | 66.95% | 88.81% | 39.52% |
| 2 | 500,000 | 64.55% | 64.55% | 88.42% | 39.24% |
| 3 | 750,000 | 69.31% | 63.49% | 88.43% | 39.11% |
| 4 | 1,000,000 | 71.97% | 62.90% | 88.49% | 39.05% |

학습 정확도는 계속 올랐지만 validation Top-1과 Macro-F1은 1 epoch 이후 계속
내려갔다. 따라서 마지막 epoch가 아니라 1 epoch EMA checkpoint를 최종 후보로
선택했다. 이는 데이터 수가 늘어난 사실과 별개로 과적합이 발생했음을 보여준다.

### 14.3 ONNX와 INT8

| artifact | 크기 | SHA-256 |
|---|---:|---|
| FP32 ONNX | 9.60MB | `00d61fa6d28aa5e2de0e8be9e5057ec20a428e14d15d31cb6d35fb72590b3e31` |
| INT8 QDQ ONNX | 2.68MB | `6c49f951c1a8af8ac6b2fae730cdaddbc4a060ea5d9607185ae0a4ba717022fc` |

- INT8는 FP32보다 72.06% 작다.
- 현재 약 1,497MB 시각 stack보다 99.82% 작다.
- 전처리 포함 단일 이미지 p50은 7.14ms로, 기존 54.15ms보다 86.82% 짧다.
- FP32 대비 INT8 Top-1 drift: -2.59%p
- FP32/INT8 Top-1 prediction agreement: 80.90%

### 14.4 외부 고정 proxy 비교와 승격 판정

평가에는 학습에 넣지 않은 Open Images 공식 validation 고정 test 3,596장을
사용했다.

| 모델 | Top-1 | Top-3 | Macro-F1 | 전처리 포함 p50 |
|---|---:|---:|---:|---:|
| 현재 운영 SigLIP 앙상블 | 68.69% | 90.66% | 51.94% | 54.15ms |
| 신규 MobileNetV4 FP32 | 64.38% | 84.65% | 36.51% | 9.09ms |
| 신규 MobileNetV4 INT8 | 61.79% | 83.01% | 34.86% | 7.14ms |

INT8 후보는 현재 운영 모델보다 Top-1 `-6.90%p`, Top-3 `-7.65%p`,
Macro-F1 `-17.08%p`였다. 양자화 전 FP32도 Top-1 `-4.31%p`,
Macro-F1 `-15.43%p`여서 양자화 방식만 바꿔 해결할 수 있는 격차가 아니다.
표본 30장 이상 leaf recall gate와 FP32 대비 양자화 drift gate도 통과하지
못했다.

결론:

- `promotion_gate_passed=false`
- `configs/cashlog/hybrid.serving.json`은 변경하지 않았다.
- 현재 운영 모델 `cashlog33-all-data-mps-v1`을 유지한다.
- 실패한 후보를 실제 서비스에 올리지 않았다.
- 사용자 실제 사진 2장은 학습 이력과 SHA-256이 겹치므로 승격 근거에서 제외했다.

### 14.5 실행 중 막힌 지점과 복구

첫 ONNX 후처리 실행은 repository 상대 경로를 만드는 코드의 `ROOT` 상수 누락으로
학습이 끝난 뒤 중단됐다. 이미 완료한 MPS 학습과 checkpoint를 다시 계산하지
않도록 `CATAI_POSTTRAIN_ONLY=true` 재개 경로를 추가했고, 동일 MLflow run에서
ONNX 변환·양자화·외부 비교를 완료했다. 최종 한국어 pipeline 로그에는 최초
실패 뒤 학습 재사용과 승격 보류가 모두 남아 있다.

### 14.6 다음 모델의 필수 변경

이번 실험 결과상 단순히 같은 작은 backbone에 더 많은 증강 뷰를 반복하는 것은
승격 근거가 되지 않는다. 다음 compact 후보는 아래를 별도 내부 validation으로
선택한 뒤, 외부 고정 test는 마지막 한 번만 사용해야 한다.

1. MobileNetV4 Medium 또는 동급 경량 backbone
2. full inverse class weight 대신 unweighted/square-root weight 비교
3. 1 epoch 이후 조기 종료 또는 더 낮은 backbone learning rate
4. 현재 SigLIP 운영 모델의 soft target을 사용하는 지식 증류
5. 양자화 민감 첫·마지막 layer를 FP32로 남기는 mixed INT8
