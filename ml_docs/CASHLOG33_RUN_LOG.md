# CashLog 33개 Leaf 실행 기록

최초 실행일: 2026-07-17 (Asia/Seoul)
최종 갱신일: 2026-07-27
분류체계: `13.33.1`, 정확히 33개 leaf ID
현재 서빙 모델: `cashlog33-all-data-mps-v1`
운영 결정: 실제 사진 검증 전까지 `guarded_integration_candidate`

## 1. 전체 결과

데이터 수집부터 학습, 평가, 후보 선정까지 이어지는 전체 DAG를 실행했다.
최초 실행에서는 메모리 제한으로 시각 헤드 학습이 한 번 실패했으나,
배치 단위 스트리밍으로 수정한 뒤 모든 작업이 완료됐다.

현재 파이프라인은 다음을 지원한다.

- CashLog의 33개 leaf ID 계약 검증
- 구성 모델과 E2E 평가 결과의 MLflow 기록
- SHA-256으로 고정된 후보 모델 생성
- Airflow 기반 데이터 및 학습 흐름 관리
- Jenkins에서 Airflow 실행을 호출하는 자동화
- 루프백 전용 MPS 모델 API에서 인증된 Top 3 추천 제공

현재 결과는 **실제 서비스 사진에서 95% 정확도를 입증한 결과가 아니다.**
동결된 수동 라벨 실제 사진 holdout이 아직 없으며, 합성 영수증, 약한 텍스트
라벨, Open Images 객체 프록시는 서로 다른 평가 범위로 관리한다.
따라서 자동 카테고리 확정은 계속 비활성화한다.

## 2. Airflow 최초 통합 실행

| 항목 | 값 |
|---|---|
| Airflow DAG | `cashlog33_training_pipeline` |
| DAG 실행 ID | `codex-20260717T0411KST` |
| 시작 | `2026-07-16T19:11:16.693054+00:00` |
| 종료 | `2026-07-16T19:26:24.352695+00:00` |
| 최종 상태 | `success` |
| 작업 결과 | 9개 중 9개 `success` |
| 재시작 후 DAG import 오류 | 0 |

작업별 결과:

| 작업 | 시도 횟수 | 결과 |
|---|---:|---|
| `validate_inputs` | 1 | 성공 |
| `build_text_dataset` | 1 | 성공 |
| `score_visual_proxy` | 1 | 성공 |
| `train_text` | 1 | 성공 |
| `train_vision_head` | 2 | 성공 |
| `build_candidate_config` | 1 | 성공 |
| `generate_e2e_fixtures` | 1 | 성공 |
| `evaluate_candidate` | 1 | 성공 |
| `select_candidate` | 1 | 성공 |

첫 번째 시각 헤드 학습은 종료 코드 137로 중단됐다. 학습기가 증강된 PIL 이미지
1,068장을 한꺼번에 메모리에 유지해 Airflow 컨테이너의 실사용 메모리 한도를
초과한 것이 원인이었다.

`scripts/train_cashlog33_vision_head.py`를 제한된 크기의 이미지 배치를 순차적으로
처리하고 각 배치를 즉시 해제하도록 수정했다. DAG 배치 크기도 16에서 4로
낮췄다. 두 번째 시도는 컨테이너 상주 메모리 약 2.08 GiB에서 완료됐다.

## 3. MLflow 실행 기록

최초 통합 실험은 experiment ID `3`, 이름 `cashlog33-hybrid-v2`에 기록됐다.

| 범위 | Run ID | 상태 |
|---|---|---|
| Airflow 텍스트 학습 | `7733ae80f3b44e59a25c68657411efcf` | `FINISHED` |
| Airflow 시각 헤드 학습 | `9307a30f940a40a988342652e71997e3` | `FINISHED` |
| Airflow 하이브리드 E2E | `139ab7ba17d84c05bc02c1c5246256c4` | `FINISHED` |
| 고정된 서빙 설정 E2E | `59a60db4e3d446fb97ab199d05b43291` | `FINISHED` |

MLflow에는 파라미터, 지표, 실행 상태, 평가 산출물을 저장한다.
서비스 주소는 `http://127.0.0.1:5500`이며 인터넷에 공개하지 않는다.

2026-07-27 전체 데이터 MPS 실행은 `cashlog33-all-data-mps` 실험에 별도로
기록했다. 해당 Run ID와 결과는 17절에 정리했다.

## 4. 최초 데이터 수집 및 범위

다음 표는 2026-07-17 최초 통합 실행 당시의 데이터 상태다.

| 소스 | 반영 결과 | 범위 | 결정 |
|---|---:|---:|---|
| 리비전 고정 MIT 합성 은행 거래 | 텍스트 18,669건 | 매핑 가능한 leaf의 약한 라벨 | 프록시 학습에 사용 |
| CashLog 결정적 템플릿 | 텍스트 15,840건 | 33/33 leaf | 라우팅 및 학습 보조 |
| Open Images V7 validation | 이미지 411장 | 23/33 leaf | 시각 객체 프록시 |
| 고정 Noto 영수증 렌더러 | 이미지 99장 | leaf당 3장 | E2E I/O 테스트 전용 |
| Openverse API | 선택 데이터 없음 | 해당 없음 | 익명 API 401/429로 당시 제외 |
| PD12M 탐색 | 선택 데이터 없음 | 해당 없음 | 데이터셋 API/index 500 계열 오류로 제외 |

최초 텍스트 데이터셋은 34,509건으로 구성됐다.

- 학습: 26,794건
- 검증: 3,839건
- 테스트: 3,876건

같은 원본이나 템플릿의 변형이 서로 다른 split으로 넘어가지 않도록 group 기반
결정적 분할을 사용했다. Open Images 수집에서는 여러 CashLog leaf에 동시에
매핑되는 모호한 이미지 4,058장을 제외했고, 승인된 이미지의 다운로드 실패는
없었다.

Open Images holdout은 의도적으로 23개 leaf만 포함한다. 월세, 공과금,
인터넷·TV, 금융 상품, 미분류 및 기타 의미는 일반 객체 라벨만으로 신뢰성 있게
추론할 수 없다. 실제 CashLog 사진이 충분히 쌓이기 전까지 해당 카테고리는
OCR·텍스트와 안전 fallback을 핵심 증거로 사용한다.

## 5. 최초 범위별 측정 결과

| 평가 범위 | 샘플 / leaf | Top-1 | Top-3 | Macro F1 |
|---|---:|---:|---:|---:|
| SigLIP2 zero-shot, Open Images 프록시 holdout | 82 / 23 | 0.5976 | 0.7073 | 0.5995 |
| 학습된 SigLIP2 선형 헤드, 동일 holdout | 82 / 23 | 0.7927 | 0.9390 | 0.8094 |
| TF-IDF SGD, 합성·약한 라벨 텍스트 테스트 | 3,876 / 33 | 0.9972 | 0.9995 | 0.9981 |
| 전체 하이브리드, 고정 Noto 합성 영수증 | 99 / 33 | 0.9899 | 0.9899 | 0.9896 |

고정 합성 하이브리드 실행의 추가 지표는 다음과 같다.

- 최소 leaf recall: `0.6667`
- ECE: `0.2743`
- 잘못된 자동확정 비율: `0.0`
- fallback 비율: `0.0404`
- 로컬 Mac p95 지연: `0.3776초`
- Airflow 컨테이너 p95 지연: `1.3453초`

모든 예측은 사용자 확인을 요구하도록 설정했다.

위 수치는 프록시와 통합 테스트 지표이며 실제 앱 정확도가 아니다.
특히 합성 텍스트 정확도는 실제보다 낙관적으로 측정될 수 있고, ECE는 이미
운영 자동확정 기준을 통과하지 못한다.

## 6. 모델 아키텍처 및 I/O 결정

추론 앙상블 구성:

1. SigLIP2 base patch16 224 임베딩 및 33개 leaf zero-shot prior
2. 고정 SigLIP2 임베딩 위에서 학습한 logistic 선형 헤드
3. 로컬에 해시 고정된 RapidOCR detector, classifier, 한국어 recognizer ONNX
4. 단어·문자 TF-IDF SGD 텍스트 분류기
5. 정규화된 CashLog OCR lexicon과 보수적인 판독 불가 fallback
6. 기존 식사 확률 안에서만 `meal_dining`과 `meal_cafe`를 재분배하는
   MobileNetV4 UECFood specialist

입력 형식:

- multipart의 `image`
- JSON의 `imageBase64`
- 지원 이미지: JPEG, PNG, WebP, HEIC, HEIF

API는 추론 전에 선언 크기, 실제 디코딩 크기, 파일 signature와 MIME,
디코딩 성공 여부, 전체 픽셀 수를 검증한다.

출력에는 분류체계 버전, 모델 버전, 추천 leaf 1개, Top 3 확률, OCR 및 근거,
fallback 사유, `need_user_check`가 포함된다.

학습 산출물은 `checkpoints/cashlog33/airflow_latest` 아래에 격리한다.
학습 작업은 `configs/cashlog/hybrid.serving.json`을 자동으로 덮어쓰지 않는다.
후보가 통합 게이트를 통과하려면 시각 모델, 시각 헤드, 텍스트 모델, OCR ONNX
3개의 SHA-256이 설정과 일치해야 한다.

## 7. 모델 선정 근거와 실패 이력

지출 카테고리 의미는 이미지 형태보다 한국어 상호명, 품목명, 거래 문구에서
드러나는 경우가 많다. 동일한 프록시 holdout에서 학습된 시각 헤드는 zero-shot
대비 Top-1을 약 19.5%p 높였다. OCR과 텍스트 경로는 일반 시각 데이터가
표현하기 어려운 카테고리를 보완한다.

거절되거나 대체된 실험:

| 시도 | 실패 또는 한계 | 처리 |
|---|---|---|
| 기존 4-leaf EfficientNet | epoch 25 이후 heartbeat timeout, 현재 분류체계와 불일치 | 과거 기록만 보존하고 미선정 |
| ConvNeXt 후보 | 호스트 메모리 압박으로 종료 코드 137 | 고정 SigLIP2 임베딩과 선형 헤드로 대체 |
| 증강 이미지 일괄 적재 | Airflow 첫 실행 종료 코드 137 | 제한된 스트리밍 배치로 수정 후 성공 |
| 초기 MLflow 산출물 | 읽기 전용 workspace artifact URI를 광고 | `mlflow-artifacts:/` 프록시 활성화 |
| 호스트 E2E MLflow 호출 1건 | sandbox가 로컬 loopback socket을 차단해 무한 재시도 | 실행 중단 후 로컬 서비스 접근 권한으로 재실행 |
| Apple 폰트 합성 fixture | Linux와 OCR 점수가 크게 달랐음 | Noto 폰트와 고정 99장 fixture로 통일 |
| 잘못된 UECFood 4-class 매핑 | `tea`가 `steak`, `steamed`에 부분 일치 | 해당 실험 폐기, 단어 경계 매칭으로 수정 |
| 전체 UECFood 미세조정 후보 | 동일 검증셋에서 기존 specialist보다 낮음 | 학습 산출물은 보존하되 서빙 모델로 미선정 |

## 8. 최초 서빙 검증

Linux ARM64용 API 이미지를 `Dockerfile.api`에서 다시 빌드했다.

- 압축 해제 Docker 크기: 592,340,355 bytes
- digest:
  `sha256:64e212fda1c2a030bc89dfdfe06407ba5c5660a8c1c2a1259cb46ef21d1041dd`
- 모델 및 체크포인트 디렉터리: read-only mount
- 런타임 uid/gid: `10001`
- root filesystem: read-only
- 임시 저장소: 용량 제한 tmpfs
- Linux capability: 전부 제거
- `no-new-privileges` 적용
- 모델 런타임: offline

`127.0.0.1:8010` 최초 검증:

| 검사 | 결과 |
|---|---|
| health 및 런타임 | HTTP 200, 하이브리드 런타임 사용 가능 |
| 내부 키 누락 | HTTP 401 |
| 유효한 키와 잘못된 Base64 이미지 | HTTP 400 |
| `X-Internal-API-Key` 호환성 | HTTP 200 |
| 기존 CashLog `X-API-Key` 호환성 | HTTP 200 |
| 실제 앱 카페 영수증 canary | `meal_cafe`, confidence 0.8143 |
| Canary Top 3 | `meal_cafe`, `meal_drink`, `meal_dining` |
| 당시 canary 분류체계/모델 | `13.33.1`, `cashlog33-hybrid-v1` |
| Canary 결정 | `need_user_check=true` |

최초 인증 요청은 모델과 OCR의 cold loading을 포함해 약 16.6초가 걸렸다.
반복 warm 요청은 약 2.0초였다. 이후 MPS native 서비스와 입력 압축을 적용한
개선 결과는 12절과 13절에 기록했다.

## 9. 운영 승격 상태와 남은 입력

현재 통합 selector의 의미는 다음과 같다.

- `integration_ready=true`
- `production_eligible=false`
- `auto_confirm_enabled=false`

필수로 부족한 입력은 동결되고, 사용 동의를 받았으며, 사람이 직접 검수한 실제
사진 holdout이다.

- 총 330장 이상
- 33개 leaf마다 독립적인 사진 10장 이상
- 올바른 `leaf_id`, SHA-256, 비식별 group ID, 동의 상태, 수동 검수 상태
- 학습이나 임계값 조정에 재사용하지 않음
- 동일 이미지 및 유사 이미지의 split 간 누수 금지
- 개인정보 비식별화와 private storage 보존·삭제 정책 적용

실제 holdout이 준비되면 한 번의 동결 평가에서 다음 기준을 모두 통과해야 한다.

- Top-1 `>=0.95`
- Top-3 `>=0.95`
- Macro F1 `>=0.75`
- 모든 leaf recall `>=0.60`
- ECE `<=0.08`
- 잘못된 자동확정 비율 `<=0.02`
- p95 지연 `<=3초`

운영자가 추가로 수행해야 하는 배포 작업:

1. 회전 가능한 하나의 secret을 worker의 `CATAI_INTERNAL_API_KEY`와 CashLog
   backend의 `PRODUCT_ANALYZER_API_KEY`로 설정한다.
2. 해당 secret을 React Native 또는 `VITE_` 변수에 넣지 않는다.
3. Jenkins credential `airflow-local-basic`과 `Jenkinsfile` 기반 Pipeline을
   생성한다.
4. 실제 Backend/Home Server를 Tailnet node로 추가하고 문서화된 ACL을 적용한다.
5. 로컬 Airflow bootstrap 계정 `admin/admin`을 loopback 외부 사용 전에
   반드시 교체한다.

## 10. Galaxy 배포 검증

2026-07-17 Galaxy의 `~/services/cashlog-gateway`에 private relay를 배포했다.
기기는 Android/aarch64와 Termux Python 3.13.13을 사용한다. 휴대폰의 CashLog
앱 저장소는 수정하지 않았으며 relay는 별도의 서비스 디렉터리와 Python 환경을
사용한다.

기존 relay에는 실제 인증 결함이 있었다. 들어온 gateway key는 전달했지만
Mac 모델 worker가 요구하는 `X-Internal-API-Key`를 추가하지 않았다.
그 결과 `/health`는 200이어도 실제 추론은 upstream 401로 실패할 수 있었다.

수정된 relay는 다음을 적용한다.

- 서로 다른 gateway key와 model API key 전달
- 요청 body streaming 및 14 MiB 제한
- JSON 응답 2 MiB 제한
- redirect 비활성화
- access log 비활성화
- health 검사에도 gateway key 요구

배포 검증:

| 검사 | 결과 |
|---|---|
| 전용 SSH identity | 키 로그인 성공 |
| SSH password 및 keyboard login | 비활성화, 실제 거부 확인 |
| SSH forwarding 정책 | reverse 전용, `GatewayPorts no` |
| secret 파일 | mode `600`, 값 출력 및 커밋 금지 |
| 키 누락 또는 잘못된 relay 호출 | HTTP 401 |
| 인증된 relay health | HTTP 200 |
| Galaxy에서 Mac tunnel health | HTTP 200 |
| 기존 카페 영수증 Galaxy 경유 | `meal_cafe`, confidence 0.8143 |
| warm E2E relay 지연 | 2.01초 |
| Galaxy LAN 포트 8000 및 18010 | 연결 거부 |
| Cloudflare Quick Tunnel | 명시적으로 요청된 테스트 단계에서만 복구, URL은 임시 |
| Galaxy Tailscale 전용 gateway bind | 인증 시 HTTP 200, 미인증 시 401 |
| macOS reverse-tunnel LaunchAgent | 실행 및 자동 재시작 확인 |
| 오래된 reverse tunnel | 제거, 관리되는 loopback tunnel만 유지 |
| 최종 Tailscale 이미지 E2E | `meal_cafe`, confidence 0.8143, 2.28초 |

Galaxy relay는 할당된 Tailscale IPv4 주소에만 bind한다. 같은 포트로 Galaxy의
Wi-Fi/LAN 주소에 연결하면 거부된다. macOS 사용자 LaunchAgent가 Galaxy
Tailscale 주소를 통해 reverse tunnel을 유지한다. 강제 재시작으로 SSH process가
교체된 뒤에도 모델 health가 자동으로 HTTP 200으로 복구되는 것을 확인했다.

gateway bind 주소와 선택된 loopback 모델 URL은 저장소가 아니라 소유자 전용
runtime 파일에 저장한다.

현재 테스트 단계에서는 인증된 Galaxy gateway 앞에 임시 Cloudflare Quick
Tunnel이 있고 Vercel이 해당 URL을 사용한다. React Native에는 Galaxy 주소나
gateway key를 전달하지 않는다. 운영 배포 전에는 임시 tunnel을 이름과 정책이
고정된 Cloudflare Tunnel로 교체해야 한다.

## 11. 모니터링 위치

- Airflow: `http://127.0.0.1:8080`
- 최초 DAG 실행: `codex-20260717T0411KST`
- MLflow: `http://127.0.0.1:5500`
- 최초 실험: `cashlog33-hybrid-v2`
- 전체 데이터 MPS 실험: `cashlog33-all-data-mps`
- Jenkins: 로컬 설정 후 `http://127.0.0.1:8081`
- 모델 리포트: `http://127.0.0.1:8010/report`
- 정적 리포트: `reports/cashlog33/model_report/index.html`
- 기계 판독 모델 선정 결과: `reports/cashlog33/model_selection.json`
- 모델 API 로그: `logs/model-api.jsonl`
- 프로세스 오류: `logs/model-api.error.log`

온라인 서비스에서 모니터링해야 할 지표:

- 요청 수와 오류 수
- warm 및 cold 지연
- fallback 비율
- 사용자 검토 비율
- 선택된 항목의 Top 3 순위
- leaf 및 모델 버전별 수정 비율

원본 이미지, 개인정보가 포함된 OCR 텍스트, JWT, 내부 API key는 로그에
기록하지 않는다.

## 12. MPS 정확도, 지연 및 관측성 개선 (2026-07-17)

Docker Desktop은 Apple MPS를 노출하지 않으므로 모델 worker를 Linux Docker
CPU에서 loopback 전용 macOS LaunchAgent로 이동했다.

적용 사항:

- 모델 eager load 및 warmup
- SigLIP2 FP16 MPS 추론
- MPS 시각 처리와 CPU OCR 병렬 실행
- 큰 OCR 입력 크기 제한
- 이미지 및 OCR 원문을 기록하지 않는 단계별 지연 로그

첫 번째 공격적인 OCR resize는 Top-1 95% 게이트에서 93.94%로 실패해
거절했다. OCR 최소 크기를 736px로 복구하고 큰 입력을 960px로 제한하자 고정
33-leaf 통합 결과가 Top-1 98.99%, Top-3 98.99%로 회복됐다.

| 측정 항목 | 적용 전 | 적용 후 |
|---|---:|---:|
| 33-leaf fixture p50 | 349ms | 233ms |
| 33-leaf fixture p95 | 378ms | 278ms |
| 1254px 카페 이미지 반복, 로컬 API | CPU Docker 약 710ms | native MPS 약 350ms |
| SigLIP2 시각 단계 p50 | 기록 없음 | 44ms |
| fixture OCR 단계 p50 | 기록 없음 | 224ms |

UECFood 식사 specialist는 class balancing을 sampler와 loss에 중복 적용하던
문제를 제거한 뒤 MPS에서 2 epoch 재학습했다. 고정 4,249장 검증셋에서
Top-1 98.05%, Top-3 100%를 기록했다.

이 수치의 범위는 `meal_dining`과 `meal_cafe`뿐이며 33-leaf 실제 사진 정확도의
증거가 아니다.

관측 가능 위치:

- 인증이 필요한 `GET /metrics`
- 응답 헤더 `X-Request-ID`, `X-Process-Time-Ms`
- `logs/model-api.jsonl`
- MLflow 실험 `cashlog33-mps-specialists`
- 실행별 `progress.json`, `training.jsonl`

## 13. 공개 경로 지연 개선 (2026-07-17)

수 초가 걸리던 앱 요청의 주된 원인은 모델 worker가 아니었다. 원본 Vercel
`iad1` 함수와 Quick Tunnel을 거친 2.65 MiB PNG 요청은 5.30~6.32초가
걸렸지만 같은 이미지를 로컬 MPS worker에서 처리하면 약 0.35초였다.
인증된 Quick Tunnel 직접 요청은 0.74~1.12초였다.

적용한 개선:

- analyzer와 analyzer-status 함수만 Vercel `icn1`에서 실행
- 큰 이미지를 upstream 전송 전에 최대 960px JPEG로 재인코딩
- 브라우저에서도 업로드 전 동일한 best-effort 축소 수행
- 최적화 실패 또는 크기 감소가 없으면 검증된 원본 유지

2.65 MiB canary는 Galaxy 전송 전에 약 138 KiB로 줄었고 `meal_cafe`
결과는 유지됐다.

| 공개 경로 측정 | 적용 전 | 적용 후 |
|---|---:|---:|
| 원본 2.65 MiB, 브라우저 최적화 우회 | 5.30~6.32초 | 2.98~4.10초 |
| 사전 압축 424 KiB | `iad1` 3.30~3.54초 | `icn1` 1.71~2.15초 |
| 최종 960px / 138 KiB 요청 | 측정 없음 | 0.94~1.12초 |
| 측정 요청 내부 모델 단계 | 상관관계 없음 | 0.34~0.59초 |

배포된 client bundle에는 브라우저 이미지 압축기가 포함된다. Vercel 응답은
다음 정보를 반환한다.

- `X-Cashlog-Read-Time-Ms`
- `X-Cashlog-Optimize-Time-Ms`
- `X-Cashlog-Analyzer-Time-Ms`
- `X-Cashlog-Total-Time-Ms`
- 입력 및 출력 byte 수
- Galaxy를 거쳐 `logs/model-api.jsonl`까지 전달되는 `X-Request-ID`

Vercel은 이미지, OCR 원문, secret을 포함하지 않는
`image_analysis_completed` JSON 이벤트도 기록한다.

이 결과는 현재 테스트 경로만 검증한다. 임시 Quick Tunnel은 운영 전에 이름과
정책이 고정된 tunnel로 교체해야 한다.

## 14. 동의 기반 피드백 수집 (2026-07-17)

- Top-1 승인, Top-3 대안 선택, 수동 leaf 수정 이벤트에 버전을 기록하도록 구현
- CashLog에 별도의 이미지 보관 동의 추가
- 일반 사진 저장을 학습 동의로 간주하지 않도록 분리
- Supabase client RLS를 `pending` 입력 전용으로 제한
- 이벤트 멱등성, 검토 상태, 모델·분류체계 버전, Top 3, private 이미지 참조 저장
- HMAC 비식별화, 중복 및 경로 검증, quarantine 구현
- active-learning 우선순위 점수와 33-leaf 준비도 게이트 구현
- 제한된 이미지 index와 일일 Airflow 정제 작업 추가
- 사람이 검토한 release가 승인되기 전까지 사용자 피드백 기반 자동 학습 비활성화

## 15. 최초 배포 전 어려운 샘플 라벨링 (2026-07-18)

- `http://127.0.0.1:8011`에 loopback 전용 33-leaf 라벨링 서버 추가
- revision 검사와 append-only 감사 로그를 포함한 원자적 라벨 결정 저장
- 확정, 수정, 거절, 필터, 검색, 현재 모델 Top 3 확인 기능 추가
- 이미지 추론을 반복하지 않고 기존 시각 헤드와 embedding cache를 재사용하는
  queue builder 추가
- 당시 queue: 총 411장, Top-1 불일치 36장, 불확실 330장, 확신 일치 45장
- 사람이 검토한 오류 탐색 데이터는 `train`에만 고정
- 해당 데이터는 배포 정확도의 근거로 사용하지 않음
- Python 테스트 26개, bytecode compile, JavaScript 문법, loopback/API 보안,
  desktop/mobile 레이아웃 검증 완료

## 16. 실제 데이터 격리 라벨링 및 증분 학습

### 16.1 실제 데이터 격리 라벨링 (2026-07-23)

- 동의받은 CashLog 이미지의 private 목적지로 `data/raw/cashlog33/actual` 추가
- 공개, 프록시, 합성 소스와 실제 데이터 분리
- 제한된 feedback release를 위한 멱등 importer 추가
- 경로 탈출 차단
- 이미지 재인코딩으로 EXIF 및 GPS 제거
- SHA-256 파일명과 비식별 sample ID 사용
- 파일 mode `0600`, 디렉터리 mode `0700`
- 로컬 mirror는 이미지와 manifest가 안전하게 반영된 뒤에만 원본 제거
- Supabase import는 원격 원본을 자동 삭제하지 않음
- feedback export 다음에 Airflow `materialize_actual_dataset` 작업 실행
- 거절되거나 동의받지 않은 이미지는 secure source index에 진입 불가
- `predeploy_labeler --actual`을 loopback 포트 `8012`에 추가
- 실제 데이터 전용 미검토 queue와
  `data/processed/cashlog33/actual_review/v1` 출력 사용
- 사람이 33-leaf 라벨을 확인하거나 수정하기 전까지 학습 대상에서 제외
- 승인된 실제 데이터는 `train`에만 고정
- 배포 정확도 평가는 별도의 미사용 holdout으로 수행

### 16.2 실제 라벨 2건 증분 학습 (2026-07-26)

- 사람이 승인한 실제 데이터 2건을 모두 `meal_dining`으로 export
- 실제 이미지, OCR 텍스트, identifier, prediction 파일은 Git에 커밋하지 않음
- 고정 Open Images embedding cache를 재사용
- 실제 이미지의 증강 view 8개를 SigLIP2 MPS에서 7.50초에 인코딩
- 후보 `cashlog33-hybrid-actual-v1-candidate` 학습
- MLflow experiment 3, run `ea44469a8f67442cbc214e9ca380d8d7`

고정 검증 및 테스트 결과는 기존과 같았다.

| 범위 | Top-1 | Top-3 | Macro F1 |
|---|---:|---:|---:|
| 검증 62장 | 75.81% | 96.77% | 72.60% |
| 테스트 82장 | 79.27% | 93.90% | 80.94% |
| 합성 E2E 99장 | 98.99% | 98.99% | 98.96% |

한 번의 warm 실행에서 후보 p50/p95는 243/292ms, 기존 서빙 모델은
238/262ms였다.

학습에 사용한 실제 데이터 2건의 fit은 1/2에서 2/2로 개선됐고,
`transit_car`에서 `meal_dining`으로 잘못 분류하던 사례가 수정됐다.
하지만 두 데이터 모두 학습에 사용했기 때문에 holdout 근거가 아니다.

결정: 실제 학습 예시는 수정했지만 미사용 정확도 지표가 개선되지 않았으므로
당시 서빙 모델 `cashlog33-hybrid-v1.1-fast`를 유지했다.

### 16.3 전체 병합 데이터셋 재학습 수정 (2026-07-27)

기존 embedding append 방식 대신 명시적이고 버전이 고정된 병합 데이터셋을
만들었다.

- Open Images 411장
- 사람이 검토한 실제 데이터 2장
- 총 413장
- 학습 269장, 검증 62장, 테스트 82장

`checkpoints/cashlog33/vision_head_v1/split_manifest.jsonl`의 기존 split을
그대로 유지했다. 실제 데이터 2건은 모두 사람 승인 상태이며 `train`에 고정했다.

- 병합 manifest SHA-256:
  `40e4213cea8eaaac58cf445de7a5554b2641d1def717a2b2a12721ecb9ff83a5`
- 위치:
  `data/processed/cashlog33/training/incremental_v1/manifest.jsonl`
- MPS 인코딩: 증강 학습 view 1,076개, 검증 62장, 테스트 82장
- 소요 시간: 34.88초
- MLflow run: `c1e9f769ca6449758da120512e013e4a`
- 후보: `cashlog33-vision-head-merged-v1`

검증 Top-1 75.81%, Top-3 96.77%, macro-F1 72.60%와 테스트 Top-1
79.27%, Top-3 93.90%, macro-F1 80.94%는 기존과 같았다.
합성 E2E 99장도 Top-1 98.99%, Top-3 98.99%, macro-F1 98.96%로 같았고,
실제 학습 데이터 2건은 2/2를 기록했다.

결정: 알려진 학습 예시는 학습했지만 미사용 지표가 개선되지 않아 이 시점에는
서빙 모델을 교체하지 않았다.

## 17. 제한 없는 전체 데이터 MPS 재학습 (2026-07-27)

### 17.1 사용 데이터

텍스트 소스의 임의 상한을 제거했다.

- 지출 leaf에 매핑 가능한 원본 60,000건 전체 사용
- 프로젝트 생성 텍스트 15,840건 전체 사용
- 총 75,840건
- 학습 59,685건
- 검증 8,067건
- 테스트 8,088건
- 원본의 나머지 8,000건은 수입 또는 이체 데이터로, 지출 전용 33개 leaf에
  유효한 정답이 없어 제외

일반 시각 manifest:

- Open Images 411장 전체
- 라이선스를 확인한 약한 Openverse 데이터 61장 전체
- 사람이 승인한 실제 데이터 2장 전체
- 원본 이미지 총 474장
- 약한 라벨은 `train` 전용
- 고정 프록시 검증 62장과 테스트 82장은 변경하지 않음

UECFood specialist:

- 전체 31,395장
- `meal_dining` 29,002장
- `meal_cafe` 2,393장
- 학습 26,686장
- 결정적 검증 4,709장
- per-class sample cap 없음
- balanced sampler를 사용하지 않아 한 epoch에 전체 학습 split을 순회

학습 가능한 전체 이미지 수는 31,869장이다.

### 17.2 텍스트 모델 결과

- manifest: 75,840건
- MLflow run: `7f6249fb766141de8c9487c8aacfd9d6`
- 테스트 Top-1: `0.9984`
- 테스트 Top-3: `1.0000`
- 테스트 macro-F1: `0.9983`

TF-IDF와 SGD는 PyTorch 연산이 아니므로 MPS 대상이 아니다. 해당 학습은 CPU에서
수행하며, 이미지 임베딩과 specialist 학습에 MPS를 사용한다.

### 17.3 일반 시각 헤드 결과

MPS에서 증강 학습 view 1,320개와 검증·테스트 이미지 전체를 인코딩했다.

- MLflow run: `de2293b95dc748c1881b84af6becdab0`
- 테스트 Top-1: `0.8049`
- 테스트 Top-3: `0.9390`
- 테스트 macro-F1: `0.7903`

기존 Top-1 `0.7927`보다 개선됐지만 macro-F1은 `0.8094`에서 낮아졌다.
따라서 단독 지표만으로 승격하지 않고 전체 하이브리드 회귀 게이트를 추가로
적용했다.

### 17.4 UECFood 라벨 오류 발견 및 수정

기존 override의 부분 문자열 비교에서 `tea`가 `steak`, `steamed` 내부에도
일치했다. 그 결과 793장이 `meal_drink`로 잘못 매핑됐다. 또한 egg,
vegetable, tofu 같은 식재료 단어를 사용해 조리된 음식 2,276장을
`meal_grocery`로 잘못 만든 문제가 있었다.

해당 4-class 실험은 전부 폐기했다.

수정 사항:

- keyword에 정규식 단어 경계 적용
- UECFood 범위를 prepared food로 명시
- UECFood에서는 신뢰할 수 있는 dining/cafe 의미만 학습
- grocery/drink는 일반 시각, OCR, 텍스트 헤드가 담당

### 17.5 식사 specialist 비교

수정된 동일 검증셋에서 기존 checkpoint를 먼저 재평가했다.

- MLflow baseline run: `ba17031e68274ec0a56357e78a4e05bd`
- Top-1: `0.9620`
- Top-3: `1.0000`
- Macro F1: `0.8865`
- 최소 leaf recall: `0.9610`

전체 26,686장 학습 split을 MPS에서 한 epoch 모두 순회한 미세조정 후보:

- 학습 run: `c4bcdab3b4db411e96f03227409bfc65`
- 평가 run: `48ecc29e0a5d442cbddc3ab233efbe2e`
- Top-1: `0.9565`
- Top-3: `1.0000`
- Macro F1: `0.8719`
- 최소 leaf recall: `0.9443`

전체 데이터 후보는 95% 게이트를 통과했지만 기존 checkpoint보다 낮았다.
따라서 전체 데이터 학습 산출물은 보존하되 서빙에는 기존 96.20% specialist를
선택했다.

### 17.6 통합 회귀 평가 및 최종 선택

고정 합성 영수증 99장 평가:

| 모델 | MLflow run | Top-1 | Top-3 | Macro F1 | p95 |
|---|---|---:|---:|---:|---:|
| 기존 서빙 baseline | `a639c571ae6242cc8e75a698208a2a42` | 0.9899 | 0.9899 | 0.9896 | 274.6ms |
| 전체 데이터 후보 | `d4fbefd6292443c2989be88cb70d0853` | 0.9899 | 0.9899 | 0.9896 | 256.7ms |

최종 선택:

- 모델 버전: `cashlog33-all-data-mps-v1`
- 시각 헤드: 전체 474장 기반 모델
- 텍스트 헤드: 전체 75,840건 기반 모델
- 식사 specialist: 수정 검증셋에서 더 강한 기존 96.20% checkpoint
- `allow_auto_confirm=false`

합성·프록시 결과는 동결된 실제 CashLog 사진에서 95% 정확도를 입증하지 않는다.

### 17.7 서빙 검증

loopback 전용 macOS LaunchAgent를 새 설정으로 재시작했다.

`/health` 확인 결과:

- `status=ok`
- `model_device=mps`
- `model_loaded=true`
- `model_version=cashlog33-all-data-mps-v1`

인증된 multipart 식재료 fixture 추론 결과:

- 정답 및 추천: `meal_grocery`
- 모델: `cashlog33-all-data-mps-v1`
- 엔진: `siglip2+mobilenetv4+rapidocr+tfidf`
- `need_user_check=true`

전체 Python 테스트는 48개가 통과했고, 2개 subtest도 통과했다.
Python bytecode compile, shell 문법 검사, `git diff --check`도 통과했다.

## 18. OCR 및 33개 leaf 데이터 10만 장 확장

### 18.1 수집 원칙

대규모 무단 스크래핑 대신 공식 API와 공식 bulk 경로를 우선했다.

- CORD v2: 고정 revision의 Hugging Face Hub API로 1,000장 수집
- Product Opener: Open Food/Beauty/Pet Food Facts API로 391장 수집
- Open Food Facts 대량 확장: product export 및 AWS image/OCR bucket 사용
- CashLog 합성 데이터: versioned text manifest에서 결정적으로 생성
- CORD의 인도네시아 domain label은 CashLog 카테고리 정답으로 사용하지 않음
- Product Opener product type은 약한 라벨이며 train-only

### 18.2 생성 및 검증 결과

- 33-leaf 합성 full-page: 112,200장
- 학습 full-page: 105,600장, leaf별 정확히 3,200장
- 합성 OCR line box: 999,194개
- CORD: 1,000장, OCR word box 23,912개
- OCR recognition crop: 130,000장, 학습 120,000장
- 전체 생성·수집 이미지 파일: 243,200장
- 고유 full-page 합성 해시: 112,200개
- 누락 파일, split 누수, box 범위 오류: 0

검증 보고서:

- `reports/cashlog33/data/ocr_category_v1_validation.json`
- `reports/cashlog33/data/ocr_category_v1_ocr_metrics.json`

### 18.3 RapidOCR 기준선

합성 검증 99장에 현재 서빙 OCR을 실행했다.

- MLflow run: `fcba681fbc3f47c8bf039082fd3feb20`
- CER: `0.1422`
- 완전 일치율: `0.1313`
- OCR 텍스트 기반 33-leaf Top-1: `0.9091`
- p50: `0.2461초`
- p95: `0.2985초`

이는 새 OCR 학습 데이터의 난이도를 측정한 기준선이다. 새 OCR checkpoint가
아직 기존 모델보다 낫다는 뜻은 아니므로 RapidOCR 모델은 교체하지 않았다.

### 18.4 상품 API 통합 학습

Product Opener API 수집 391장 중 동일 이미지가 서로 다른 leaf에 걸린 2개
row를 SHA-256 기준으로 제외했다. 최종 389장은 학습 전용 약한 라벨이다.

- 통합 manifest: 863장
- train/validation/test: 719/62/82
- MPS embedding 및 선형 헤드 학습: 59.51초
- MLflow run: `af0c36f69862457daf64831dce6c93b2`

기존 모델 대비 고정 proxy test 결과:

- Top-1: `0.8049 → 0.8293`
- Top-3: `0.9390 → 0.9512`
- Macro-F1: `0.7903 → 0.8067`

고정 validation은 Top-1 `0.7742 → 0.7581`, Macro-F1
`0.7412 → 0.7253`으로 하락했다. 테스트 개선과 검증 회귀가 함께 있으므로
후보 checkpoint와 MLflow 산출물은 보존하되 운영 모델 자동 교체는 보류했다.

### 18.5 자동화

- 로컬 실행 job: `cashlog_ocr_category_112k`
- Airflow: CORD/API provenance 확인 → 생성 → 전수 검증 → OCR 기준선 평가
- MLflow: OCR CER·지연·카테고리 지표와 후보 모델 artifact 저장
- 오류 시 기존 승인 manifest와 운영 모델을 덮어쓰지 않음

최종 검증에서 Python 테스트 53개와 subtest 2개가 통과했다. 새 Python 파일의
bytecode compile, shell 문법 검사, `git diff --check`도 통과했다.

## 19. 50만 장 과적합 완화 학습

### 19.1 데이터 설계

v1의 반복적인 종이 영수증 형태와 직접적인 `분류 메모 <정답>` 힌트를 제거했다.
v2는 다음 변형을 결정적으로 생성한다.

- receipt, statement, mobile, invoice 네 가지 layout
- Apple SD Gothic Neo, Apple Gothic, Apple Myungjo 세 가지 한글 font
- 원근, 밝기, 대비, blur, noise, downsample, motion blur
- clean/light/medium OCR text corruption
- source text group의 train/validation/test 고정

생성 결과:

- 전체 full-page: 508,200장, 12.01GiB
- train: 501,600장, leaf별 정확히 15,200장
- validation/test: 각각 3,300장
- OCR line: 4,434,123개
- 고유 이미지 SHA-256: 508,200개
- 기존 데이터와 합친 text train: 561,285건
- source group leakage: 0

초기 전수 검증에서 원근 변환 반올림으로 이미지 밖에 걸친 OCR box가 3개
발견됐다. 검증기를 50만 row 스트리밍 방식으로 바꾸고, 6개 샘플의 12개
좌표를 이미지 경계로 보정했다. 재검증 결과 오류는 0이다.

### 19.2 Product Opener 추가 수집

- 신규 API 수집: 581장
- 두 API run 합계: 972 row
- 동일 product ID 제거: 292 row
- 교차 leaf 동일 이미지 제거: 2 row
- 최종 unique train-only 약한 라벨: 678장
- 식재료 신규 검색: HTTP 503, 기존 98장 보존

MPS 시각 후보는 고정 proxy test에서 Top-1 `0.8293 → 0.8415`,
Macro-F1 `0.8067 → 0.8183`으로 올랐다. validation Top-1은 `0.7581`로
기존 운영 시각 헤드의 `0.7742`보다 낮아 승격하지 않았다.
MLflow run: `9e892db140df4b989824fdccd4c7f393`.

### 19.3 텍스트 후보 비교

신규 v2-only noisy test 3,300건에서 기존 모델:

- Top-1: `0.9200`
- Top-3: `0.9333`
- Macro-F1: `0.9081`
- ECE: `0.0554`

561,285건 전체를 학습한 `alpha=3e-5` 후보는 Top-1 `0.9367`이었지만 기존
v1 RapidOCR 고정셋에서 회귀해 제외했다.

최종 선정한 `alpha=1e-5` 후보:

- MLflow training run: `da3edd5709df4503b4923c20aa4e7610`
- fit time: 175.38초
- v2-only Top-1: `0.9552`
- v2-only Top-3: `0.9655`
- v2-only Macro-F1: `0.9562`
- v2-only ECE: `0.0396`

실제 RapidOCR 출력 99장 비교:

| 고정셋 | 기존 | 최종 후보 | 변화 |
|---|---:|---:|---:|
| v2 multi-layout | 0.9293 | 0.9596 | +3.03%p |
| v1 receipt | 0.9091 | 0.9192 | +1.01%p |

후보 config는 `configs/cashlog/hybrid.text500k-alpha1e5-candidate.json`이다.
합성 고정셋에서는 두 방향 모두 개선됐지만 실제 사용자 사진 holdout이 없으므로
`configs/cashlog/hybrid.serving.json`은 변경하지 않았다.

### 19.4 자동화와 모니터링

- 로컬 dataset job: `cashlog_ocr_category_500k`
- 로컬 training job: `cashlog_text_500k_alpha1e5`
- Airflow DAG: `cashlog33_500k_training_pipeline`
- MLflow experiment: `cashlog33-500k`
- 전체 hash, box, class balance, source-group leakage 검증이 학습 선행 조건
- 후보 DAG는 serving config를 수정하지 않는 promotion-gated 구조

최종 검증은 Python 테스트 57개와 subtest 2개가 통과했다. Python compile,
shell 문법 검사, `git diff --check`, 후보 artifact SHA-256 대조도 통과했다.

## 20. 50만 장 모델 서빙 승격

사용자 승인에 따라 최종 `alpha=1e-5` 텍스트 후보를 실제 서빙 설정에
승격했다.

- 이전 model version: `cashlog33-all-data-mps-v1`
- 신규 model version: `cashlog33-500k-mps-v2`
- 신규 text artifact:
  `checkpoints/cashlog33/text_500k_v2_alpha1e5/text_model.joblib`
- SHA-256:
  `6220594bf65cb8b1b39d45862550be56722551797634b7c66a9568d0f6f3474a`
- 시각 헤드와 meal specialist는 변경하지 않음
- `allow_auto_confirm=false` 유지

상세 출처와 증강 명세는
`ml_docs/CASHLOG33_DATA_PROVENANCE_AUGMENTATION.md`에 기록했다.

macOS LaunchAgent를 재설치하고 MPS worker를 재시작했다.

`/health` 결과:

- `status=ok`
- `model_loaded=true`
- `model_device=mps`
- `model_version=cashlog33-500k-mps-v2`
- `model_load_ms=13260.42`
- `model_warmup_ms=913.84`

보호된 runtime key를 사용한 multipart fixture 추론:

- HTTP 200
- 추천 및 정답 fixture: `health_gym`
- confidence: `0.7480`
- 응답 model: `cashlog33-500k-mps-v2`
- `need_user_check=true`

초기 신규 text weight `0.60`, lexicon weight `0.15`에서는
`edu_class` 한 건이 `misc_uncat`으로 밀려 E2E Top-1이 `0.9798`이었다.
OCR에서 `학원` lexicon이 명확히 검출됐으므로 with-lexicon fusion을
다음처럼 보정했다.

- vision: `0.25`
- text: `0.50`
- lexicon: `0.25`

99장 전체 재평가:

- MLflow run: `439e0f654dac465a9c8f2d4befc7c7f4`
- Top-1: `0.9899`
- Top-3: `0.9899`
- Macro-F1: `0.9896`
- minimum recall: `0.6667`
- `need_user_check_rate=1`
- `false_auto_confirm_rate=0`

이 값은 기존 서빙 정확도를 회복하면서 50만 장 텍스트 모델을 사용하는
최종 설정이다. 실제 사용자 사진의 동결 holdout이 생기기 전까지
`allow_auto_confirm=false`를 유지한다.

최종 fusion 반영 후 LaunchAgent를 다시 시작했다.

- `/health`: `status=ok`, `model_loaded=true`, `model_device=mps`
- model version: `cashlog33-500k-mps-v2`
- load: `12200.13ms`
- warm-up: `931.64ms`
- 회귀 확인 fixture: `edu_class/fixture-00.jpg`
- 실제 API 추천: `edu_class`
- confidence: `0.2713`
- matched lexicon: `학원`
- `need_user_check=true`

## 21. 운영자 품질 판정에 따른 즉시 롤백

실제 사용 결과가 나쁘다는 운영자 판정에 따라 50만 장 후보를 서빙에서 즉시
제거했다. 학습 artifact와 평가 보고서는 원인 분석을 위해 보존하지만 재승격은
금지한다.

- 제거: `cashlog33-500k-mps-v2`
- 복원: `cashlog33-all-data-mps-v1`
- 복원 text artifact:
  `checkpoints/cashlog33/text_all_data_v1/text_model.joblib`
- 복원 SHA-256:
  `fa85c45c9d9464116c8fe16ff02d6656960a948acf5d752647244b5639b13fba`
- 복원 with-lexicon fusion: vision `0.25`, text `0.60`, lexicon `0.15`
- `allow_auto_confirm=false` 유지

LaunchAgent 재시작 후 검증:

- `/health`: `status=ok`, `model_loaded=true`, `model_device=mps`
- model version: `cashlog33-all-data-mps-v1`
- `edu_class/fixture-00.jpg` 실제 API 추천: `edu_class`
- `need_user_check=true`

50만 장 후보는 실제 사진 동결 holdout과 운영자 검수를 통과하기 전까지 평가
전용으로 유지한다.

## 22. 실제 원본 데이터 확장과 MPS 재학습 (2026-07-29)

### 22.1 원본 수집과 검증

합성 렌더 수량이 아니라 서로 다른 실제 이미지 수를 늘렸다.

- Open Images V7 validation human label: 17,983장
- Amazon Berkeley Objects: 23,272장
- 기존 actual/Open Products/Openverse train-only 원본: 741장
- 최종 추가 train-only 원본: 24,013장
- 전체 원본 감사: 43,290 row, 고유 SHA-256 42,995개
- 출처 간 중복 hash: 294개
- cross-leaf 충돌: 1개

ABO에서는 147,702 listing을 읽어 main image 기준 29,941개를 매핑했다.
leaf cap 이후 24,122개를 archive에서 검증했고, 최소 변 96px 미만 또는 decode
실패 850개를 제외했다. image ID가 여러 leaf에 매핑된 296개와 동일 이미지를
공유한 listing 5,892개도 제거했다.

### 22.2 I/O와 누수 방지

입력:

- `data/raw/cashlog33/openimages_v7/manifest.jsonl`
- `data/processed/cashlog33/training/originals_v2_additional/manifest.jsonl`
- 각 row의 `relative_path`, `leaf_id`, `source`, `sha256`, license/attribution

출력:

- `checkpoints/cashlog33/vision_head_originals_v2/embedding_cache.npz`
- `checkpoints/cashlog33/vision_head_originals_v2/vision_head.joblib`
- `checkpoints/cashlog33/vision_head_originals_v2/vision_head_weighted.joblib`
- `checkpoints/cashlog33/vision_head_originals_v2/vision_head_ensemble.joblib`
- `reports/cashlog33/originals_v2/`

Open Images만 결정적 train/validation/test split으로 사용했다. ABO, actual,
Open Products, Openverse는 `split_lock=train`으로 고정했다. 평가셋에는
추가 원본을 넣지 않았고 SHA-256 중복·충돌을 통합 단계에서 차단했다.

### 22.3 학습과 막힌 지점

첫 실행은 Codex sandbox에서 PyTorch가 MPS를 사용할 수 없어 중단됐다. native
실행으로 전환한 뒤 MPS가 정상 활성화됐다.

- Open Images train: 11,691장, 4-view 46,764개
- validation: 2,696장
- test: 3,596장
- 추가 원본: 24,013장, 4-view 96,052개
- 총 train augmented sample: 142,816개
- embedding 시간: 2,948.70초

학습 완료 후 MLflow 3.14가 file tracking store를 기본 차단해 마지막 기록
단계가 실패했다. 생성된 406MB embedding cache와 모델 artifact는 정상
보존됐으므로 이미지 재인코딩 없이 source weight와 앙상블을 튜닝했다.
최종 결과는 기존 MLflow 서버 `http://127.0.0.1:5500`에 다시 기록했다.

### 22.4 모델 선택

신규 head 단독 비교:

- Top-1: `0.7358 → 0.8665`
- Top-3: `0.9221 → 0.9855`
- Macro-F1: `0.5427 → 0.7253`
- 실패 원인: `meal_grocery -37.50%p`, `life_goods -66.67%p`,
  `health_med -25.00%p`

additional source weight를 `0.10`으로 조정해 퇴행 폭을 줄였고, validation에서
2%p leaf regression gate를 처음 통과하는 최소 신규 head 비중 `0.10`을
선택했다. 최종 모델은 기존 head 90%와 source-weighted 신규 head 10%의 확률
앙상블이다.

고정 test 결과:

- Top-1: `0.7358 → 0.7781` (+4.23%p)
- Top-3: `0.9221 → 0.9511` (+2.89%p)
- Macro-F1: `0.5427 → 0.5791` (+3.64%p)
- 23개 평가 leaf 최소 recall 변화: `0.0000`
- proxy gate: 통과
- production gate: 차단

실제 CashLog 사람이 라벨링한 동결 holdout은 2장뿐이다. 따라서 운영 모델
`cashlog33-all-data-mps-v1`과
`configs/cashlog/hybrid.serving.json`은 변경하지 않았다.

MLflow:

- experiment: `cashlog33-original-data`
- final run ID: `3754ae4163a042e4a71f06ffd795d533`

## 23. 100만 뷰 경량 모델 MPS 학습과 승격 보류 (2026-07-30)

### 23.1 수집·학습 입력

- Open Images V7 train human-verified 후보: 469,006장
- 라이선스·다운로드·디코딩 검증 통과: 469,001장
- 12MB 원본 제한으로 영구 제외: 5장
- 기존 원본: 24,013장
- 전체 고유 원본: 493,014장
- train 원본: 399,213장
- validation 원본: 93,801장
- 23개 시각 leaf에 사용 가능한 고유 train 원본: 399,198장
- 외부 고정 test와 SHA-256/source ID 중복: 0건

모든 사용 가능한 train 원본을 최소 한 번 포함한 뒤 원본이 적은 leaf를
water-filling했다. 정확히 1,000,000개 `원본 index + augmentation seed` 뷰를
만들었으며, 100만 개를 고유 원본 수로 기록하지 않는다.

증강:

- RandomResizedCrop 0.65~1.00
- RandomHorizontalFlip p=0.5
- ColorJitter 0.20/0.20/0.15/0.02
- RandAugment 2회, magnitude 7
- RandomErasing p=0.10

### 23.2 MPS·MLflow 실행

- backbone: `mobilenetv4_conv_small`
- device: `mps`
- batch: 128
- 총 처리 views: 1,000,000
- MLflow experiment: `cashlog33-million-compact`
- run ID: `cf9524b498104f02b9749d7493357466`

validation Macro-F1 기준 best는 epoch 1이었다.

| epoch | 누적 views | train Top-1 | validation Top-1 | Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 250,000 | 60.46% | 66.95% | 39.52% |
| 2 | 500,000 | 64.55% | 64.55% | 39.24% |
| 3 | 750,000 | 69.31% | 63.49% | 39.11% |
| 4 | 1,000,000 | 71.97% | 62.90% | 39.05% |

학습 정확도 상승과 validation 하락이 동시에 나타나 과적합으로 판정했다.

### 23.3 ONNX·양자화

- FP32 ONNX: 9.60MB
- 정적 INT8 QDQ ONNX: 2.68MB
- FP32 대비 크기 감소: 72.06%
- 현재 시각 stack 대비 크기 감소: 99.82%
- 전처리 포함 단일 이미지 p50: `54.15ms → 7.14ms`
- FP32 대비 INT8 Top-1 drift: -2.59%p

첫 후처리 실행은 경로 helper의 `ROOT` 상수 누락으로 중단됐다. checkpoint를
재학습하지 않고 `CATAI_POSTTRAIN_ONLY=true`로 동일 MLflow run을 재개해
ONNX 내보내기부터 비교까지 완료했다.

### 23.4 고정 외부 test와 모델 교체 판정

학습에 사용하지 않은 Open Images 공식 validation 고정 3,596장 비교:

| 모델 | Top-1 | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| 현재 운영 모델 | 68.69% | 90.66% | 51.94% |
| 신규 FP32 | 64.38% | 84.65% | 36.51% |
| 신규 INT8 | 61.79% | 83.01% | 34.86% |

신규 INT8는 작고 빨랐지만 정확도·Macro-F1·leaf recall·양자화 drift gate를
통과하지 못했다. FP32부터 기존보다 낮아 양자화 조정만으로 해결할 수 있는
상태도 아니었다.

- 최종 gate: 실패
- serving config 교체: 없음
- 유지 모델: `cashlog33-all-data-mps-v1`
- 실제 사용자 사진 2장은 학습 이력과 중복되므로 gate에서 제외

상세 출처, split, 증강, I/O, checksum, leaf별 결과는
`ml_docs/CASHLOG33_MILLION_COMPACT_V1.md`와
`reports/cashlog33/million_v1/compact_comparison.json`에 기록했다.

## 24. 동일 SigLIP2 데이터 확장·혼합 INT8 검증 (2026-07-30)

이전 MobileNetV4 실험은 backbone이 달라 데이터 추가 효과를 현재 운영
SigLIP2와 직접 비교할 수 없었다. 비교 계약을 바로잡아 현재와 같은
SigLIP2 Base patch16 224, 같은 전처리, 같은 0.30/0.70 vision blend, 같은
meal specialist를 사용하고 학습 데이터만 추가했다.

- 신규 Open Images 공식 train 원본: 469,001장
- 최종 head 학습 embeddings: 611,817개
- 고정 외부 test: 3,596장
- MPS float16 embedding 처리량: 39.20장/초
- embedding 시간: 12,005.15초
- MLflow training run: `1c04395494bb4e8d9e77e7247787a4f5`
- MLflow monitor run: `be6b9da8b0b74663bcde9f274c5aac85`

동일 serving 경로 결과:

| 모델 | Top-1 | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| 현재 SigLIP2 | 68.69% | 90.66% | 51.94% |
| 추가 데이터 SigLIP2 native | 75.78% | 93.97% | 62.47% |
| 같은 후보 mixed INT8 | 75.39% | 93.91% | 62.63% |

전체 dynamic INT8은 97.11MB였지만 정확도 손실이 컸다. encoder 앞 6개
block을 FP32로 보호하고 뒤 6개 block만 QInt8로 변환한 218.36MB mixed
후보는 native 대비 Top-1 drift를 -0.39%p로 줄였다.

최종 승격은 보류했다. `meal_grocery` recall이 현재 대비 native -10.23%p,
mixed INT8 -11.36%p였고 mixed INT8 p50 93.05ms가 현재 MPS p50 54.15ms보다
느렸다. 결과 확인 후 gate를 완화하지 않았으며 serving config는 변경하지 않았다.
상세 기록은 `ml_docs/CASHLOG33_SAME_SIGLIP_EXPANSION.md`와
`reports/cashlog33/siglip_expanded_v1/final_comparison.json`이다.
