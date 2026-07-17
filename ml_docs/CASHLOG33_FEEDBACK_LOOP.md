# CashLog33 Feedback Loop

## 목적

현재 단계는 정책경사 기반 강화학습이 아니라 사용자 확정값을 정답으로 축적하는
human-in-the-loop 능동학습이다. 33개 leaf 분류 문제에는 이 방식이 직접적이며,
검수되지 않은 사용자 입력으로 모델이 오염되는 것을 막을 수 있다.

## 데이터 흐름

```text
CashLog 사진 저장
  -> Supabase cashlog_category_feedback (pending, RLS)
  -> Airflow cashlog_feedback_curation (매일)
  -> HMAC 비식별 export + 잘못된 행 quarantine
  -> 운영자 이미지/라벨 검수
  -> approved 행만 능동학습 후보 선별
  -> MLflow 지표/산출물 기록
  -> 33 leaf 데이터 게이트
  -> 명시적 학습 승인
```

일반 사진 보관 동의와 모델 개선용 이미지 보관 동의는 분리한다. 사용자가 모델
개선 동의를 하지 않아도 확정 카테고리 메타데이터는 품질 통계에 사용할 수 있지만,
이미지 경로는 학습 큐로 내보내지 않는다.

## 앱 이벤트

사진 분석 뒤 기록을 저장하면 모델 결과를 그대로 선택한 경우까지 이벤트를 남긴다.

- `accepted_prediction`: 모델 Top-1을 확정
- `top3_selection`: Top-1이 아닌 Top-3 후보를 확정
- `manual_edit`: Top-3 밖의 leaf를 직접 확정

이벤트에는 임의 UUID, 모델 버전, taxonomy 버전, Top-3, 확정 leaf, 시각, 별도 이미지
동의를 기록한다. OCR 원문, 메모, 금액, 사용자 이메일은 피드백 이벤트에 넣지 않는다.

## Supabase 보안 경계

마이그레이션: CashLog 저장소의
`supabase/migrations/202607170001_active_learning_feedback.sql`

- 사용자는 자신의 `pending` 이벤트만 삽입할 수 있다.
- 사용자는 자신의 이벤트를 조회하거나 삭제할 수 있다.
- 사용자는 `approved`, `rejected`, `reviewed_at`을 직접 설정할 수 없다.
- 동의 이미지 경로는 반드시 해당 사용자 비공개 폴더로 시작해야 한다.
- `event_id` unique index로 네트워크 재시도 중복을 막는다.
- Service Role Key는 Airflow와 운영 검수 프로세스에만 둔다.

## Airflow와 MLflow

`cashlog_feedback_curation` DAG는 매일 실행된다.

1. Supabase와 HMAC 비밀값 존재 여부를 fail-closed로 확인한다.
2. 필요한 열만 읽고 사용자 ID를 HMAC-SHA256 그룹 ID로 치환한다.
3. 일반 이벤트 파일에서 사용자 ID, expense ID, request ID, 이미지 경로를 제거한다.
4. 승인된 이미지 경로만 권한 `0600`의 `secure_image_index.jsonl`에 분리한다.
5. 스키마 위반, 경로 소유권 위반, UUID 중복을 `quarantine.jsonl`로 격리한다.
6. 승인 데이터의 오분류, 낮은 margin, 낮은 confidence, leaf 부족도를 조합해 우선순위를 계산한다.
7. 요약 지표와 후보 파일을 MLflow `cashlog33-feedback-curation` 실험에 기록한다.

각 Airflow run은 고유 릴리스 경로를 사용하며, 완성된 파일 세트가 있는 재시도는 같은
스냅샷을 재사용한다. 일부 파일만 남은 불완전 릴리스는 자동 덮어쓰지 않고 실패시켜
운영자가 상태를 확인하게 한다.

Airflow 컨테이너 환경에만 다음 비밀값을 주입한다.

```bash
SUPABASE_URL=https://PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
CASHLOG_FEEDBACK_HMAC_KEY=...
```

HMAC 키는 API 키와 별도로 생성하고 32바이트 이상이어야 한다. 키가 바뀌면 동일
사용자의 그룹 ID도 바뀌므로 데이터 릴리스 기간 중에는 회전 계획을 따로 관리한다.

## I/O 계약

| 파일 | 포함 정보 | 접근 |
|---|---|---|
| `events.jsonl` | 비식별 피드백, Top-3, 확정 leaf, 모델/taxonomy, 검수 상태 | ML 운영자 |
| `secure_image_index.jsonl` | 승인된 event/sample과 비공개 object key 매핑 | 최소 권한 운영자, `0600` |
| `quarantine.jsonl` | 행 번호, event ID, 오류 사유만 포함 | ML 운영자 |
| `training_candidates.jsonl` | 승인·동의된 표본과 능동학습 우선순위 | 학습 파이프라인 |
| `approved_metadata_feedback.jsonl` | 승인된 이미지/비이미지 피드백 | 품질·보정 분석 |
| `curation_summary.json` | 수정률, Top-3 포함률, 모델/leaf별 누적과 게이트 | Airflow, MLflow |
| `per_leaf_feedback.csv` | 33 leaf별 승인/이미지/수정 수 | Airflow, MLflow |

일반 파일에는 원본 사용자 ID나 object key가 없어야 한다. `group_id`는 데이터 분할
누출 방지를 위한 HMAC 값이며 사용자를 복원하는 식별자로 사용하지 않는다.

## 검수와 승인

검수자는 비공개 Supabase Storage에서 이미지를 확인하고 PII, 잘못된 라벨, 중복,
촬영 품질을 판단한다. 승인 파일 예시는 다음과 같다.

```json
{"event_id":"00000000-0000-4000-8000-000000000001","decision":"approved"}
{"event_id":"00000000-0000-4000-8000-000000000002","decision":"rejected"}
```

적용:

```bash
python scripts/review_cashlog_feedback.py --decisions review-decisions.jsonl
```

검수되지 않은 `pending` 행은 학습 후보가 되지 않는다. 승인 후 다음 Airflow 실행에서
새 릴리스에 포함된다.

## 학습 게이트

기본 게이트는 각 33개 leaf마다 승인·이미지 동의 표본 10개 이상이다. 추가로 실제
학습 전에는 다음을 확인한다.

- taxonomy가 `13.33.1`과 일치한다.
- 중복 sample/event와 경로 소유권 위반이 없다.
- 이미지 PII 검수와 라벨 검수를 통과했다.
- 동일 HMAC `group_id`가 train/validation/test에 걸쳐 섞이지 않는다.
- frozen real-photo holdout은 학습 후보와 분리한다.

`ready_for_training=true`는 수량 게이트 통과만 의미한다. 현재
`auto_training_allowed=false`이며 운영자의 릴리스 승인 없이 학습 DAG를 자동 호출하지
않는다. 사용자 삭제나 동의 철회가 발생하면 이후 export에서 제외하고, 아직 보관 중인
과거 릴리스와 학습 전 staging 사본도 함께 삭제해야 한다.
