# CashLog 50만 장 과적합 완화 데이터 및 학습

작성일: 2026-07-27

## 목표

기존의 작은 proxy holdout과 반복적인 합성 template에 대한 과적합을 줄이기
위해 33개 leaf 균형 학습 이미지를 50만 장 이상으로 늘리고, 기존 데이터를
버리지 않은 채 OCR 텍스트 카테고리 모델을 다시 학습한다.

## 데이터 I/O

입력:

- `data/processed/cashlog33/text/all_v1/manifest.jsonl`
- `configs/cashlog/categories.json`
- Product Opener API manifests
- CORD v2 OCR-only manifest

주요 출력:

- 이미지: `data/processed/cashlog33/ocr_category/v2_500k`
- OCR text: `data/processed/cashlog33/ocr_text/v2_500k`
- 통합 text: `data/processed/cashlog33/text/all_500k_v2`
- 최종 후보: `checkpoints/cashlog33/text_500k_v2_alpha1e5`
- 검증 보고서: `reports/cashlog33/data/ocr_category_v2_500k_validation.json`
- 비교 보고서: `reports/cashlog33/500k`

대용량 이미지와 checkpoint는 Git에 저장하지 않는다. 생성기, source hash,
검증 보고서, 후보 checksum, 실행 문서만 버전 관리한다.

## 과적합 방지

1. 33개 leaf별 train 수를 15,200장으로 고정한다.
2. source text group이 split을 넘지 못하게 한다.
3. 정답 카테고리 이름을 직접 출력하던 hint를 제거한다.
4. 네 layout, 세 font, 일곱 종류의 촬영 열화를 조합한다.
5. OCR 문자열에 clean/light/medium 오류를 결정적으로 적용한다.
6. 신규 API 이미지는 train-only 약한 라벨로 제한한다.
7. 기존 v1과 신규 v2 holdout을 모두 통과해야 후보로 선정한다.
8. 실제 사용자 사진 holdout 없이는 운영 모델을 자동 교체하지 않는다.

## 최종 규모

| 구성 | train | validation | test |
|---|---:|---:|---:|
| v2 full-page/OCR text | 501,600 | 3,300 | 3,300 |
| 기존+v2 text 통합 | 561,285 | 11,367 | 11,388 |

- v2 full-page 전체: 508,200장
- OCR line: 4,434,123개
- 고유 이미지 hash: 508,200개
- source group leakage: 0
- 최종 전수 검증 오류: 0

## 모델 비교

| 모델 | v2 Top-1 | v2 Top-3 | v2 Macro-F1 | v2 ECE |
|---|---:|---:|---:|---:|
| 기존 text model | 0.9200 | 0.9333 | 0.9081 | 0.0554 |
| 500k `alpha=3e-5` | 0.9367 | 0.9561 | 0.9384 | 0.1379 |
| 500k `alpha=1e-5` | **0.9552** | **0.9655** | **0.9562** | **0.0396** |

최종 후보는 `alpha=1e-5`다. 실제 RapidOCR 출력에서도 v2 Top-1은
`0.9293 → 0.9596`, v1 Top-1은 `0.9091 → 0.9192`로 개선됐다.

## 실행

```bash
.venv/bin/python scripts/launch_training.py cashlog_ocr_category_500k
tail -f logs/cashlog_ocr_category_500k.log
```

Airflow에서는 `cashlog33_500k_training_pipeline`, MLflow에서는
`cashlog33-500k` experiment를 사용한다.

## 선정 상태

후보 모델과 checksum-pinned config 생성까지 완료했다. 합성·proxy 지표는
개선됐지만 실제 CashLog 사용자 사진의 동결 holdout이 없으므로 운영 serving
config는 교체하지 않았다. 다음 승격 조건은 33개 leaf 실제 사진 holdout에서
기존 모델 대비 Top-1, Macro-F1, 최소 leaf recall 회귀가 없는 것이다.
