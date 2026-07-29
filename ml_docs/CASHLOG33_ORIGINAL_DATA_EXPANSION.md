# CashLog33 실제 원본 데이터 확장 v2

## 1. 목적

50만 장 후보는 63,528개 합성 text group에서 508,200개 문서를 렌더링한
OCR-text 데이터였다. 합성 proxy에서는 개선됐지만 실제 사용 품질이 나빠
롤백됐다. 이번 작업은 합성 이미지 수가 아니라 서로 다른 실제 원본 이미지와
출처 다양성을 늘리는 것이 목적이다.

운영 모델 `cashlog33-all-data-mps-v1`은 이번 작업에서 변경하지 않는다. 신규
모델은 실제 CashLog 동결 holdout을 통과하기 전까지 후보로만 관리한다.

## 2. 추가 출처

### Open Images V7

- 제공자: Google Open Images
- 사용 범위: human-verified validation image labels
- 원본 후보: 17,983장
- license: annotation CC BY 4.0, 각 선택 이미지 CC BY 2.0
- 저장 메타데이터: 원본 URL, landing URL, 저자, 저자 URL, title, license URL,
  source label, SHA-256
- mapping 범위: 23개 CashLog leaf
- 제외: 여러 CashLog leaf에 동시에 매핑되는 이미지 4,058개

Open Images 자체도 원본 이미지의 license 상태를 보증하지 않는다고 명시한다.
따라서 수집 시 각 row의 image metadata가 CC BY 2.0인지 다시 확인하고,
attribution 정보를 manifest에 고정한다.

### Amazon Berkeley Objects

- 제공자: Amazon.com
- 전체 규모: 147,702 listing, 398,212 unique catalog images
- 사용 archive: 256px catalog image archive
- license: archive 내 `LICENSE-CC-BY-4.0.txt` 기준 CC BY 4.0
- mapping 전 후보: 29,941개 main image
- leaf당 5,000개 cap 적용 후 archive 검증 후보: 24,122개
- 최종 학습 가능 원본: 23,272장
- 제외: 최소 변 96px 미만 또는 decode 실패 850장
- mapping 범위: 14개 CashLog leaf
- 한 image가 여러 leaf로 해석되면 제외
- 동일 main image를 공유하는 listing은 image ID 기준 병합
- 상품별 main image 한 장만 사용

AWS Registry에는 BY-NC 4.0으로 표시되지만 공식 ABO download page와 실제
download archive의 license 파일은 BY 4.0이다. 재현 가능한 판단을 위해 실제
archive license 파일과 그 SHA-256을 source artifact에 함께 보존한다.

## 3. 의미 매핑 원칙

상품이나 물체가 보인다고 지출 의도가 자동으로 정해지지는 않는다. 다음 원칙을
적용한다.

- 직접적인 상품 유형만 weak label로 사용한다.
- `CELLULAR_PHONE_CASE`를 `comm_mobile`로 사용하지 않는다.
- 포장 coffee/tea/cookie/cake를 `meal_cafe`로 사용하지 않는다.
- jewelry를 `gift_present`로 사용하지 않는다.
- `SPORTING_GOODS`처럼 gym/hobby가 모호한 상위 타입은 제외한다.
- rent, utility, fee, insurance처럼 문맥이 필요한 leaf는 상품 이미지로 만들지
  않는다.

상세 mapping은 `configs/cashlog/abo_product_type_mapping.json`에 고정한다.

## 4. split과 누수 방지

- Open Images: 결정적 source split으로 train/validation/test를 만든다.
- ABO: 전량 `split_lock=train`으로 사용한다.
- 실제 CashLog 사람이 검수한 데이터: train-only 또는 별도 real holdout 정책을
  따른다.
- 동일 SHA-256이 여러 출처에 있으면 한 장만 사용한다.
- 같은 SHA-256이 서로 다른 leaf에 있으면 양쪽 모두 격리한다.
- 상품의 multiple view를 서로 다른 split에 배치하지 않는다.
- 합성 데이터와 실제 원본 데이터의 평가지표를 합산하지 않는다.

## 5. 학습 구조

1. Open Images 전체를 SigLIP2로 score한다.
2. Open Images 고정 split의 embedding을 만든다.
3. ABO와 사람이 검수한 실제 사진은 train-only embedding으로 추가한다.
4. Logistic Regression vision head의 `C`를 validation Macro-F1로 선택한다.
5. ABO domain 영향도를 additional source weight로 조정한다.
6. 현재 head 90%, 신규 head 10%의 확률 앙상블을 validation gate에서 선택한다.
7. 같은 Open Images test embedding에서 현재 운영 head와 앙상블을 비교한다.
8. 결과와 artifact를 MLflow `cashlog33-original-data` experiment에 기록한다.

Apple Silicon에서는 MPS를 사용한다. 실행 명령:

```bash
.venv/bin/python scripts/launch_training.py cashlog_originals_v2_mps
```

로그:

```bash
tail -f logs/cashlog_originals_v2_mps.log
```

## 6. 승격 조건

proxy gate:

- 후보 Top-1이 현재 head보다 높아야 한다.
- 후보 Macro-F1이 현재 head보다 높아야 한다.
- leaf recall 하락이 2%p를 넘으면 안 된다.
- source별 지표를 별도로 기록한다.

production gate:

- 33개 leaf의 사람이 라벨링한 실제 CashLog 동결 holdout이 필요하다.
- 운영자 육안 검수를 통과해야 한다.
- `allow_auto_confirm=false`를 유지한다.
- proxy gate만 통과한 모델은 서빙에 올리지 않는다.

## 7. 현재 한계

- 공개 object/product label은 실제 지출 맥락의 정답이 아니다.
- housing, finance, telecom fee 계열은 실제 영수증·청구서 원본이 필요하다.
- 실제 CashLog 사람이 검수한 원본은 아직 2장뿐이다.
- 따라서 이번 데이터 확장은 visual component 개선 단계이며 전체 33-leaf
  서비스 정확도를 증명하지 않는다.

## 8. 실행 결과

원본 감사:

- 전체 row: 43,290개
- 고유 SHA-256: 42,995개
- Open Images 고정 proxy: 17,983장
- 추가 train-only 원본: 24,013장
- 추가 원본 중 ABO: 23,272장
- train augmented view: 142,816개
- embedding 시간: 2,948.70초

신규 head를 단독 사용하면 Top-1과 Macro-F1은 상승했지만 일부 leaf가 크게
퇴행했다. additional source weight를 `0.10`으로 낮춘 뒤에도 `health_med`,
`life_goods` 퇴행이 남았다. 따라서 validation에서 leaf recall 하락 2%p 이내를
처음 만족하는 보수적 확률 앙상블을 선택했다.

고정 Open Images test 3,596장 결과:

| 지표 | 현재 head | 최종 앙상블 | 변화 |
|---|---:|---:|---:|
| Top-1 | 0.7358 | 0.7781 | +0.0423 |
| Top-3 | 0.9221 | 0.9511 | +0.0289 |
| Macro-F1 | 0.5427 | 0.5791 | +0.0364 |
| 최소 leaf recall 변화 | - | 0.0000 | 퇴행 없음 |

proxy gate는 통과했다. 그러나 실제 CashLog 수동 holdout이 2장뿐이므로
production gate는 차단했고 `configs/cashlog/hybrid.serving.json`은 변경하지
않았다.

MLflow:

- experiment: `cashlog33-original-data`
- final run: `3754ae4163a042e4a71f06ffd795d533`
- artifact: `vision_head_ensemble.joblib`
