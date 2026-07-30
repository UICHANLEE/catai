# CashLog33 동일 SigLIP2 데이터 확장·경량화 실행 기록

## 1. 실험 목적

이 실험은 기존 운영 모델보다 작은 다른 backbone을 새로 학습하는 실험이 아니다.
현재 운영 중인 SigLIP2 구조를 고정하고 학습 데이터만 추가했을 때 성능이
개선되는지 확인한 뒤, 같은 후보를 ONNX INT8로 경량화해 최종 승격 여부를
판정한다.

비교 순서는 다음과 같다.

1. 현재 운영 SigLIP2 모델
2. 같은 SigLIP2와 추가 데이터로 재학습한 native 후보
3. 2번 후보를 그대로 변환한 ONNX FP32
4. 3번을 weight-only INT8로 양자화한 ONNX 후보

이전 MobileNetV4 100만 view 실험은 데이터 확장과 architecture 변경이 동시에
일어나 현재 운영 SigLIP2와의 직접적인 데이터 추가 효과 비교로 사용할 수 없다.
해당 결과는 경량 architecture 탐색 실패 실험으로만 보존한다.

## 2. 고정한 모델 계약

- vision backbone: `google/siglip2-base-patch16-224`
- 입력 크기: RGB 224x224
- 입력 정규화: channel별 mean 0.5, std 0.5
- feature: L2 정규화된 SigLIP2 image embedding
- 학습 head: scikit-learn multinomial `LogisticRegression`
- head regularization: `C=3.0`
- class weight: `balanced`
- vision 확률 혼합: zero-shot 0.30 + linear head 0.70
- meal specialist: 기존 모델과 같은 checkpoint, blend 0.65
- OCR·text·lexicon: 기존 serving config와 동일
- taxonomy: CashLog 33개 leaf와 기존 API I/O 계약 유지

현재 모델과 후보의 architecture, 전처리, 출력 순서, 후처리, 고정 test는
같다. 학습용 image embedding 생성만 Apple MPS float16으로 가속하며, 검증과
운영 비교는 기존 고정 test embedding 및 실제 serving 경로로 수행한다.

## 3. 데이터

### 3.1 신규 원본

- Open Images V7 공식 train 원본: 469,001장
- 후보 469,006장 중 다운로드·디코딩·크기 검증 실패: 5장
- augmentation: 없음. 신규 원본을 각각 정확히 1회 사용한다.
- 공식 Open Images validation 원본을 신규 학습 입력에 넣지 않는다.

### 3.2 기존 학습 입력

- 기존 고정 base train embedding: 46,764 views
- 기존 추가 원본의 4-view embedding: 96,052 views
- 신규 Open Images 공식 train: 469,001 originals/views
- 최종 head 학습 입력 예정: 611,817 embeddings

신규 원본에 4-view 증강을 적용하지 않은 이유는 100만이라는 숫자를 만들기 위해
같은 원본을 반복하는 대신 모든 실제 원본을 한 번씩 포함하고, 데이터 추가 효과를
더 명확하게 측정하기 위해서다. 기존 cache의 96,052 views는 현재 운영 head가
사용했던 기존 학습 이력을 보존하기 위해 그대로 포함한다.

### 3.3 누수·중복 검사

469,001개 신규 row 전체에 대해 다음 조건을 사전 검사했다.

- 파일 존재와 decode 완료 manifest
- taxonomy leaf 유효성
- 신규 manifest 내부 sample ID 중복 없음
- 신규 manifest 내부 SHA-256 중복 없음
- 기존 base validation/test와 SHA-256 중복 없음
- 기존 Open Images validation과 source image ID 중복 없음
- official split이 `train`인지 확인

검사 결과 신규 469,001장이 통과했고 고정 test와 확인된 중복은 0건이다.

## 4. 학습 실행

- device: Apple MPS
- embedding dtype: float16
- batch size: 128
- 신규 embedding shard: 4,096 originals
- 총 shard: 115
- shard 원자적 저장과 sample ID 순서 검증
- 중단 후 기존 shard 재사용 가능
- 한국어 진행 로그:
  `checkpoints/cashlog33/vision_head_siglip_expanded_v1/embedding_shards/progress_ko.jsonl`
- 전체 학습 로그:
  `logs/cashlog33-siglip-expanded-v1/training.log`
- MLflow experiment: `cashlog33-same-siglip-expanded`
- MLflow monitor run:
  `be6b9da8b0b74663bcde9f274c5aac85`

## 5. 평가 계약

학습에 사용하지 않은 Open Images 공식 validation 고정 test 3,596장을 현재
모델과 후보 모두에 사용한다.

현재 운영 baseline:

| 모델 | Top-1 | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| 현재 SigLIP2 serving | 68.6874% | 90.6563% | 51.9388% |

승격은 다음 조건을 모두 만족해야 한다.

- native 후보 Top-1이 현재 모델보다 높다.
- native 후보 Macro-F1이 현재 모델보다 높다.
- INT8 후보 Top-1이 현재 모델보다 높다.
- INT8 후보 Macro-F1이 현재 모델보다 높다.
- 표본 30장 이상 leaf의 recall 최대 하락이 10%p 이내다.
- INT8 Top-1 하락이 native 대비 0.5%p 이내다.
- INT8 Macro-F1 하락이 native 대비 1.0%p 이내다.
- INT8 artifact가 현재 vision stack보다 작다.
- INT8 단일 이미지 p50이 현재 vision 경로보다 빠르다.

실제 사용자 사진 2장은 이미 학습 이력과 중복되므로 승격 근거에서 제외한다.

## 6. ONNX I/O와 양자화

ONNX에는 SigLIP2 vision encoder, 학습된 LogisticRegression head, 고정된
zero-shot text prompt embedding, 0.30/0.70 확률 혼합을 함께 넣는다. text
encoder는 prompt embedding으로 동결되므로 ONNX runtime에서 다시 로드하지
않지만 수학적인 vision 출력은 같은 구조다.

```text
input
  name: pixel_values
  dtype: float32
  shape: [N, 3, 224, 224]
  resize: bicubic direct resize 224x224
  normalization: (rgb / 255 - 0.5) / 0.5

output
  name: vision_scores
  dtype: float32
  shape: [N, 33]
  kind: normalized probabilities
  label order: labels.json
```

양자화:

- ONNX opset 18
- dynamic mixed-precision weight-only quantization
- encoder 0~5 block FP32 보호, encoder 6~11 block MatMul/Gemm weight QInt8
- per-channel
- ONNX Runtime CPUExecutionProvider

운영 모델을 사용한 변환 smoke test:

| 항목 | 결과 |
|---|---:|
| FP32 ONNX | 355.06MB |
| INT8 ONNX | 97.11MB |
| native↔FP32 Top-1 agreement | 100% |
| FP32↔INT8 Top-1 agreement | 100% |
| FP32↔INT8 mean absolute error | 0.01507 |
| INT8 model-only single image p50 | 44.67ms |

위 agreement는 2장 변환 smoke test 결과이며 성능 판정 수치가 아니다. 최종
후보는 고정 3,596장 전체를 다시 평가한다.

## 7. 최종 결과

### 7.1 학습

- 최종 head 입력: 611,817 embeddings
- 신규 원본: 469,001
- 신규 embedding 생성 시간: 12,005.15초
- shard 기준 생성 시간: 11,965.21초
- 평균 처리량: 39.20 originals/s
- validation: Top-1 78.49%, Top-3 96.48%, Macro-F1 60.75%
- 고정 test head 단독: Top-1 78.84%, Top-3 97.02%, Macro-F1 64.93%
- native head SHA-256:
  `b46b7aade7fe066a6bbff7addf8a8a8a7d6009c2dea0e397a3f757f06fe23621`
- MLflow training run:
  `1c04395494bb4e8d9e77e7247787a4f5`
- MLflow progress monitor run:
  `be6b9da8b0b74663bcde9f274c5aac85`

### 7.2 동일 serving 경로 비교

| 모델 | Top-1 | Top-3 | Macro-F1 | vision p50 |
|---|---:|---:|---:|---:|
| 현재 SigLIP2 | 68.69% | 90.66% | 51.94% | 54.15ms |
| 추가 데이터 SigLIP2 native | 75.78% | 93.97% | 62.47% | 78.37ms |
| 같은 후보 mixed INT8 ONNX | 75.39% | 93.91% | 62.63% | 93.05ms |

native 후보는 현재 모델 대비 Top-1 +7.09%p, Top-3 +3.31%p,
Macro-F1 +10.53%p다. mixed INT8 후보도 현재 모델보다 높고 native 대비
Top-1 drift는 -0.39%p로 허용 범위 안이다.

그러나 표본 88장인 `meal_grocery` recall은 현재 75.00%에서 native 64.77%,
mixed INT8 63.64%로 각각 10.23%p, 11.36%p 하락했다. 또한 mixed INT8의
CPU p50은 93.05ms로 현재 MPS p50 54.15ms보다 느렸다.

### 7.3 양자화 선택

전체 encoder INT8은 97.11MB였지만 3,596장 Top-1이 70.66%까지 하락했다.
마지막 encoder block FP32 보호와 첫 encoder block FP32 보호를 각각 탐색했다.
첫 6개 block을 FP32로 보호하는 후보가 256장 FP32↔INT8 Top-1 agreement
100%로 가장 작은 안정 후보였다.

- FP32 ONNX: 355.06MB
- mixed INT8 ONNX: 218.36MB
- 크기 감소: 38.50%
- FP32 SHA-256:
  `9037a71981b44469a5695446662426aa838724d84bcee71fbbc386ddbda1d261`
- mixed INT8 SHA-256:
  `662155c45b4ef68e146663d004f31d4c7c36b5595d2ffa0224606b53cb9539d7`
- 256장 native↔FP32 Top-1 agreement: 100%
- 256장 FP32↔mixed INT8 Top-1 agreement: 100%
- CoreML NeuralNetwork provider p50: 140.56ms
- CoreML MLProgram: ONNX Conv pad 변환 오류로 compile 실패

### 7.4 최종 결정

- 동일 architecture 데이터 추가 효과: 확인
- native 정확도 개선: 확인
- mixed INT8 정확도 보존: 확인
- leaf recall 보호: 실패
- 추론 속도 개선: 실패
- 최종 promotion gate: 실패
- `configs/cashlog/hybrid.serving.json`: 변경하지 않음
- 유지 운영 모델: `cashlog33-all-data-mps-v1`

전체 지표가 좋아졌다는 이유로 사전에 정한 leaf recall과 latency gate를 결과 확인
후 완화하지 않았다. 후보와 ONNX artifact는 다음 개선 실험의 baseline으로
보존한다.

## 8. 산출물

- native head:
  `checkpoints/cashlog33/vision_head_siglip_expanded_v1/vision_head.joblib`
- native metrics:
  `checkpoints/cashlog33/vision_head_siglip_expanded_v1/metrics.json`
- ONNX:
  `checkpoints/cashlog33/vision_head_siglip_expanded_v1/onnx/`
- native serving 평가:
  `reports/cashlog33/siglip_expanded_v1/candidate_serving_visual.json`
- INT8 serving 평가:
  `reports/cashlog33/siglip_expanded_v1/int8_serving_visual.json`
- 최종 비교와 gate:
  `reports/cashlog33/siglip_expanded_v1/final_comparison.json`
