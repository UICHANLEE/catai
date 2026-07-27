# CashLog 데이터 출처 및 증강 명세

작성일: 2026-07-27
대상 모델: `cashlog33-500k-mps-v2`
분류체계: `13.33.1`, 33개 leaf

## 1. 목적과 해석 범위

이 문서는 CashLog 학습 데이터가 어디에서 왔고, 어떤 라벨로 변환됐으며,
50만 장 데이터에 어떤 이미지·문자열 증강이 적용됐는지 재현 가능하게 기록한다.

학습 데이터 50만 장이 실제 사용자 사진 50만 장이라는 뜻은 아니다. v2의
508,200장은 프로젝트가 생성한 합성 문서이며 실제 사진 정확도를 증명하지
않는다. 실제 사진으로 취급되는 데이터는 CORD와 외부 상품 이미지이고, 이들도
CashLog 33개 leaf의 강한 정답으로 사용하지 않는다.

## 2. 원천 데이터

| 출처 | 확보 방식 | 수량 | 라이선스·상태 | CashLog 사용 방식 |
|---|---|---:|---|---|
| US bank transaction categories v2 | revision 고정 Hugging Face file API | 매핑 60,000행 | MIT, 원천 자체가 합성 거래 | 33개 leaf 약한 텍스트 라벨 |
| CashLog text templates | 로컬 결정적 생성기 | 15,840행 | 프로젝트 생성 | 33개 leaf 통합·학습 보조 |
| CashLog OCR v2 | 위 텍스트를 문서 이미지로 렌더링 | 508,200장 | 프로젝트 생성 | OCR와 OCR-text category |
| CORD v2 | revision 고정 Hugging Face Hub API | 1,000장 | CC BY 4.0 | OCR box/layout만 사용, category 금지 |
| Product Opener 계열 | 공식 Open Food/Beauty/Pet Food Facts API | unique 678장 | 이미지 CC BY-SA 3.0 metadata | product type 약한 라벨, train-only |
| Open Images V7 | 공식 metadata와 image URL | 411장 | annotation CC BY 4.0, image별 CC BY 2.0 | object-to-expense proxy |
| Openverse | 공식 API | 61장 | CC BY/CC0/PDM | query 기반 약한 라벨, train-only |
| UECFood256 | 제공 archive | 31,395장 | 재배포 라이선스 불명확 | dining/cafe specialist만 로컬 학습 |
| CashLog actual | 사용자 동의와 수동 승인 | 2장 | 비공개 | 강한 라벨, train-only |

### Product Opener 수집 결과

두 API run에서 972 row를 받았다. 통합 시 동일 product ID 292개와 서로 다른
leaf에 걸린 동일 이미지 2개를 제외해 678장을 남겼다.

- 식재료: 98장
- 음료: 198장
- 미용: 198장
- 반려동물: 184장

두 번째 식재료 검색은 HTTP 503으로 실패했다. 제한을 우회하지 않고 기존
98장을 보존했으며, 대량 확장은 공식 product export와 AWS image/OCR bucket을
사용하도록 데이터 source catalog에 기록했다.

## 3. 50만 장 구성

| split | leaf당 | 전체 |
|---|---:|---:|
| train | 15,200 | 501,600 |
| validation | 100 | 3,300 |
| test | 100 | 3,300 |
| 합계 | 15,400 | 508,200 |

기존 텍스트 manifest와 v2를 합친 최종 텍스트 모델 입력은 다음과 같다.

- train: 561,285행
- validation: 11,367행
- test: 11,388행
- source group: 63,528개
- split을 넘은 source group: 0개

## 4. 이미지 증강

모든 난수는 `seed=500027`, split, leaf, index로 결정된다. 같은 코드·입력·font
환경에서는 같은 sample ID와 렌더링 조건을 재현한다.

### 문서 형태

각 샘플은 동일 확률로 다음 네 형태 중 하나를 사용한다.

- `receipt`: 세로 종이 영수증과 미세한 종이 줄무늬
- `statement`: 상단 색상 band와 표 구분선이 있는 거래명세서
- `mobile`: 어두운 배경과 밝은 글자의 모바일 결제 화면
- `invoice`: 따뜻한 용지색과 section header가 있는 invoice

### Font와 크기

- Apple SD Gothic Neo
- Apple Gothic
- Apple Myungjo
- 본문 20~28px, 제목 28~34px
- image width: 448, 480, 512, 544 중 하나
- image height: 640, 704, 768 중 하나

### 촬영·압축 열화

| 변형 | 적용 |
|---|---|
| 원근 변환 | 변당 최대 짧은 축의 3.5% 이동 |
| 대비 | `0.72~1.22` |
| 밝기 | `0.80~1.12` |
| Gaussian blur | 35%, radius `0.25~1.15` |
| Gaussian noise | 20%, sigma `1.0~4.5` |
| 저해상도 왕복 resize | 18%, 원본 폭의 `45~75%` |
| 수평 motion blur | 10%, kernel 3/5/7 |
| JPEG 품질 | 68~91 |

원근 변환은 이미지와 OCR polygon에 같은 행렬을 적용한다. 초기 전수 검사에서
반올림으로 경계를 벗어난 좌표를 발견해 생성기에서 모든 point를 이미지 경계로
clip하도록 수정했다. 최종 데이터에서 수정된 것은 6개 sample의 12개 point다.

## 5. OCR 문자열 증강

OCR 인식 오류에 대한 텍스트 분류기의 민감도를 줄이기 위해 이미지의 정답
transcription에서 별도의 noisy text manifest를 생성한다.

| 수준 | 비율 | 처리 |
|---|---:|---|
| clean | 약 20% | 원문 유지 |
| light | 약 45% | 낮은 확률의 문자 누락·치환, 공백 반복 |
| medium | 약 35% | 더 높은 누락·치환, line drop, 구두점 손실 |

실제 생성 결과:

- clean: 101,380행
- light: 228,541행
- medium: 178,279행

대표 치환은 `0/O`, `1/I`, `5/S`, `8/B`와 일부 한글 OCR 혼동이다. 변형은
sample ID에서 파생한 seed로 결정되어 재현 가능하다.

## 6. 라벨 누수 방지

- v1에 존재하던 `분류 메모 <정답 카테고리명>` 출력을 v2에서 금지했다.
- 원천 `group_key`를 split lock으로 사용한다.
- 동일 SHA-256 이미지가 다른 leaf에 걸리면 양쪽 모두 제외한다.
- API와 Openverse label은 validation/test에 들어가지 않는다.
- CORD semantic category는 CashLog leaf로 변환하지 않는다.
- 실제 holdout label은 모델 예측으로 갱신할 수 없다.

## 7. 검증 결과

- 이미지: 508,200장
- 고유 SHA-256: 508,200개
- OCR line: 4,434,123개
- 누락 이미지: 0
- manifest/OCR ID 불일치: 0
- 이미지 밖 OCR point: 0
- leaf 불균형: 0
- source-group leakage: 0

전수 검증 보고서는
`reports/cashlog33/data/ocr_category_v2_500k_validation.json`이다.

## 8. 학습과 모델 선정

최종 텍스트 모델은 기존 59,685개 train row와 v2 501,600개 train row를 모두
사용한다.

- model family: word/character TF-IDF + SGD log loss
- alpha: `1e-5`
- max iteration: 100
- fit time: 175.38초
- artifact: `checkpoints/cashlog33/text_500k_v2_alpha1e5/text_model.joblib`
- SHA-256: `6220594bf65cb8b1b39d45862550be56722551797634b7c66a9568d0f6f3474a`
- MLflow run: `da3edd5709df4503b4923c20aa4e7610`

고정 v2 noisy holdout Top-1은 `0.9200 → 0.9552`, Macro-F1은
`0.9081 → 0.9562`로 개선됐다. 실제 RapidOCR 출력 99장에서도 v2
`0.9293 → 0.9596`, v1 `0.9091 → 0.9192`로 개선됐다.

의미 단어가 검출된 경우의 최종 서빙 fusion은 vision `0.25`, text
`0.50`, lexicon `0.25`이다. 99장 hybrid E2E 재평가 결과 Top-1
`0.9899`, Macro-F1 `0.9896`, false auto-confirm `0`을 기록했다.
MLflow run은 `439e0f654dac465a9c8f2d4befc7c7f4`이다.

## 9. 한계

- 실제 CashLog 사용자 사진의 동결 holdout은 아직 없다.
- 합성 문서의 95.96%는 운영 정확도로 표현할 수 없다.
- 33개 leaf 중 실제 사진이 거의 없는 category가 남아 있다.
- Product Opener label은 상품 유형이지 소비 맥락의 확정 정답이 아니다.
- OCR detector/recognizer 자체는 이번 50만 장으로 미세조정하지 않았다.
  이번 승격 대상은 OCR 결과를 category로 변환하는 텍스트 모델이다.
