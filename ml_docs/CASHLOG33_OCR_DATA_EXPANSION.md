# CashLog OCR 및 33개 카테고리 데이터 확장

작성일: 2026-07-27
분류체계: `13.33.1`, 33개 leaf
상태: 데이터 생성 및 무결성 검증 완료, 실제 사진 운영 정확도 미검증

## 1. 목표와 결론

목표는 다음 두 학습 문제에 각각 10만 장 이상의 입력을 준비하는 것이다.

1. 영수증과 거래 문서의 글자 위치 및 문자열을 읽는 OCR
2. OCR 텍스트와 문서 이미지를 CashLog의 33개 leaf로 분류

확보 결과:

| 데이터 | 학습 | 검증 | 테스트 | 용도 |
|---|---:|---:|---:|---|
| CashLog 합성 full-page | 105,600 | 3,300 | 3,300 | OCR detection, 33-leaf 문서 분류 |
| CashLog 합성 line crop | 120,000 | 5,000 | 5,000 | OCR recognition |
| CORD 실제 영수증 | 800 | 100 | 100 | 실제 촬영 OCR 및 영수증 layout |

합성 full-page는 leaf마다 학습 3,200장으로 정확히 균형을 맞췄다.
CORD는 인도네시아 영수증이므로 CashLog 카테고리 정답으로 사용하지 않고 OCR
학습에만 사용한다.

## 2. 데이터 소스 조사

### CORD v2

- 원본: NAVER CLOVA CORD v2
- 수집 경로: Hugging Face Hub API의 리비전 고정 미러
- 고정 리비전: `51b7e932f2a07a883d77487b34aa5a33f76fa713`
- 라이선스: CC-BY-4.0
- 이미지: 1,000장
- OCR word box: 23,912개
- 용도: OCR detection, recognition, receipt layout
- 카테고리 학습: 금지

관련 문서:

- `https://github.com/clovaai/cord`
- `https://huggingface.co/datasets/naver-clova-ix/cord-v2`

### Open Food Facts 계열

조사 대상:

- Open Food Facts: 식재료 및 음료
- Open Beauty Facts: 미용·화장품
- Open Pet Food Facts: 반려동물
- Open Products Facts: 생활용품 및 일반 상품

공식 Product Opener API는 소량 검색에 사용할 수 있지만, 읽기와 검색에 rate
limit이 있다. 수백 건 이상과 대량 이미지는 API 반복 호출 대신 공식 product
export와 AWS Open Data를 사용하라고 명시한다.

라이선스:

- 상품 database: ODbL
- 개별 database content: Database Contents License
- 상품 이미지: CC-BY-SA

공식 문서:

- `https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/`
- `https://openfoodfacts.github.io/openfoodfacts-server/api/aws-images-dataset/`
- `https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/tutorials/scanning-cosmetics-pet-food-and-other-products/`

bounded API 수집기를 구현했지만 실행 시 Product Opener가 반복적으로 HTTP 503을
반환했다. rate limit 우회나 무한 재시도는 하지 않는다. 실패는 summary에
기록하고 다음 대량 확장에서는 공식 export와 AWS 이미지 bucket을 사용한다.

### PaddleOCR 학습 형식

PaddleOCR의 detection 학습은 이미지 경로와 `points`, `transcription` 배열을
사용한다. recognition 학습은 잘린 line 이미지 경로와 정답 문자열을 tab으로
구분한다.

현재 데이터는 두 형식을 모두 생성한다.

- detection: `paddle_det_train.txt`
- recognition: `rec_train.txt`

현재 서빙 OCR은 RapidOCR의 한국어 PP-OCRv5 ONNX 모델이다. ONNX 파일은 직접
미세조정하는 학습 checkpoint가 아니다. 새 데이터로 PaddleOCR 계열을 학습한
뒤 ONNX로 export하고, 동일 고정 평가에서 기존 모델보다 좋아졌을 때만 교체한다.

## 3. 합성 데이터 설계

입력 텍스트는
`data/processed/cashlog33/text/all_v1/manifest.jsonl`의 버전 고정 데이터를
사용한다.

생성 과정:

1. 원본 split과 leaf를 유지한다.
2. 거래일시, 승인번호, 공급가액, 부가세, 결제금액을 추가한다.
3. 한글과 영문 상호·품목·거래 문구를 여러 줄로 배치한다.
4. 글자별이 아니라 line별 사각형 좌표와 정확한 전사를 저장한다.
5. 원근 변화, 명암, 밝기, blur, noise, JPEG 품질 변화를 적용한다.
6. 변환 행렬을 OCR box에도 동일하게 적용한다.
7. 이미지와 원본 텍스트 manifest의 SHA-256을 저장한다.

합성 영수증에 카테고리 이름을 항상 직접 쓰지 않는다. 실제 거래 문구를 주요
증거로 사용하고 일부 샘플에만 메모 형태로 표시한다.

합성 데이터는 실제 영수증과 동일한 분포가 아니므로 실제 정확도 지표로
사용하지 않는다.

## 4. 파일 I/O

### Full-page 데이터

루트:

`data/processed/cashlog33/ocr_category/v1`

주요 파일:

- `manifest.jsonl`: 이미지, leaf, split, 해시, 크기, 출처
- `ocr_annotations.jsonl`: 전체 전사와 line별 box
- `paddle_det_train.txt`
- `paddle_det_validation.txt`
- `paddle_det_test.txt`
- `summary.json`
- `images/<split>/<leaf_id>/*.jpg`

### Recognition line crop

루트:

`data/processed/cashlog33/ocr_recognition/v1`

주요 파일:

- `rec_train.txt`
- `rec_validation.txt`
- `rec_test.txt`
- `manifest_train.jsonl`
- `manifest_validation.jsonl`
- `manifest_test.jsonl`
- `summary.json`
- `images/<split>/*.jpg`

### CORD

루트:

`data/raw/cashlog33/cord_v2`

주요 파일:

- `manifest.jsonl`
- `ocr_annotations.jsonl`
- `summary.json`
- `images/<split>/*.jpg`

원본 이미지와 대용량 manifest는 `.gitignore` 대상이다. Git에는 수집기, 생성기,
검증기, 설정과 요약 보고서만 저장한다.

## 5. 실행 방법

전체 수집·생성·검증:

```bash
.venv/bin/python scripts/launch_training.py cashlog_ocr_category_112k
tail -f logs/cashlog_ocr_category_112k.log
```

직접 실행:

```bash
.venv/bin/python scripts/collect_cashlog_cord.py
.venv/bin/python scripts/generate_cashlog33_ocr_category_dataset.py --workers 8
.venv/bin/python scripts/export_cashlog33_ocr_recognition_crops.py
.venv/bin/python scripts/validate_cashlog33_ocr_category_dataset.py \
  --verify-all-hashes
```

현재 OCR 품질 평가:

```bash
.venv/bin/python scripts/evaluate_cashlog33_ocr_dataset.py \
  --per-leaf 3 \
  --device mps \
  --mlflow-tracking-uri http://127.0.0.1:5500
```

## 6. 검증 결과

전체 해시 검증 결과:

- 합성 full-page: 112,200장
- 학습 full-page: 105,600장
- 고유 이미지 해시: 112,200개
- 합성 OCR line box: 999,194개
- CORD: 1,000장
- CORD OCR word box: 23,912개
- 누락 이미지: 0
- manifest와 OCR ID 불일치: 0
- 이미지 밖 OCR point: 0
- 33개 leaf 누락: 0
- leaf별 학습 이미지: 정확히 3,200장

현재 RapidOCR의 합성 검증 99장 결과:

- CER: `0.1422`
- 완전 일치율: `0.1313`
- OCR 텍스트 기반 33-leaf Top-1: `0.9091`
- p50: `0.2461초`
- p95: `0.2985초`
- MLflow run: `fcba681fbc3f47c8bf039082fd3feb20`

이 결과는 OCR이 새 변형을 읽지만 개선 여지가 있다는 의미다. 합성 데이터에서
목표를 맞춘 뒤에도 실제 영수증 holdout에서 별도로 평가해야 한다.

## 7. 모델 교체 기준

OCR 후보는 다음 조건을 모두 만족할 때만 서빙 모델을 교체한다.

- 고정 합성 검증에서 기존 CER보다 낮음
- CORD test에서 기존 CER보다 낮음
- 실제 한국어 CashLog 영수증 holdout에서 회귀 없음
- 33-leaf 하이브리드 Top-1과 macro-F1 회귀 없음
- p95 지연이 운영 기준 이내
- ONNX artifact와 문자 dictionary의 SHA-256 고정

현재 `korean_PP-OCRv5_rec_mobile.onnx`는 그대로 유지한다. 새 데이터가
추가됐다는 이유만으로 검증되지 않은 OCR 모델을 자동 승격하지 않는다.

## 8. 상품 API 이미지 통합 결과

Product Opener 계열 API에서 다음 이미지를 수집했다.

| source | 수집 | 충돌 제거 후 |
|---|---:|---:|
| Open Food Facts 식재료 | 99 | 98 |
| Open Food Facts 음료 | 100 | 99 |
| Open Beauty Facts | 100 | 100 |
| Open Pet Food Facts | 92 | 92 |
| 합계 | 391 | 389 |

동일 SHA-256 이미지가 식재료와 음료에 동시에 매핑된 2개 row는 어느 쪽도
정답으로 간주하지 않고 제외했다. API의 product type은 사람이 승인한 CashLog
정답이 아니므로 모든 이미지는 약한 라벨이자 학습 전용이다.

고정 검증·테스트 split을 유지하고 MPS에서 SigLIP2 선형 헤드를 다시 학습했다.

| 모델 | 검증 Top-1 | 검증 Macro-F1 | 테스트 Top-1 | 테스트 Top-3 | 테스트 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| 기존 `all_data_v1` | 0.7742 | 0.7412 | 0.8049 | 0.9390 | 0.7903 |
| API 통합 후보 `all_data_v2` | 0.7581 | 0.7253 | 0.8293 | 0.9512 | 0.8067 |

- 후보 checkpoint: `checkpoints/cashlog33/vision_head_all_data_v2`
- MLflow run: `af0c36f69862457daf64831dce6c93b2`
- 테스트 Top-1 변화: `+2.44%p`
- 테스트 Macro-F1 변화: `+1.64%p`
- 검증 Top-1 변화: `-1.61%p`

테스트 지표는 개선됐지만 검증 지표가 하락했으므로 운영 모델을 자동 교체하지
않았다. API 후보는 보존하고 실제 CashLog holdout과 카테고리별 오류 검토 후
승격 여부를 결정한다.
