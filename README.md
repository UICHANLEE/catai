# Catai

CashLog 영수증·상품 사진을 33개 leaf 카테고리로 추천하는 학습 및 비공개
서빙 패키지입니다.

현재 선정 모델:

- Model: `cashlog33-hybrid-v1`
- Taxonomy: `13.33.1`, 앱과 동일한 33개 leaf ID
- Architecture: `SigLIP2 + linear head + Korean RapidOCR + text SGD + lexicon`
- Serving config: `configs/cashlog/hybrid.serving.json`
- Status: `guarded_integration_candidate`, 실사진 holdout 전 자동 확정 금지

현재 리포트의 높은 텍스트·E2E 수치는 synthetic/weak proxy와 고정 렌더링
영수증에 대한 결과이며 실제 CashLog 사진 정확도가 아닙니다. 과거 패키지의
4-leaf 음식 모델은 호환용으로 남아 있지만 현재 33-leaf 서빙 후보가 아닙니다.

## 설치

```bash
cd /Users/uichan/workspace/catai
.venv/bin/python -m pip install -e .
```

33-leaf 하이브리드 추론까지 로컬에서 실행하려면 모델 의존성을 포함해
설치합니다.

```bash
.venv/bin/python -m pip install -e ".[hybrid]"
```

모델 가중치와 다운로드 데이터는 크기·라이선스·보안을 위해 Git이나 패키지
wheel에 넣지 않습니다. `Dockerfile.api`는 런타임만 만들고 `models/`와
`checkpoints/`를 읽기 전용 볼륨으로 마운트합니다.

```bash
python -m pip install "catai[hybrid] @ git+https://github.com/UICHANLEE/catai.git@250716"
```

필요한 파일과 SHA-256은 `configs/cashlog/hybrid.serving.json`에 고정되어
있습니다. 재현 가능한 데이터·모델 준비 절차는 `TRAINING.md`와
`ml_docs/CASHLOG33_DATA_CARD.md`를 따릅니다.

## CLI 예측

```bash
.venv/bin/catai-predict-cashlog path/to/product.jpg \
  --hybrid-config configs/cashlog/hybrid.serving.json \
  --device auto \
  --pretty
```

Apple Silicon에서 MPS를 강제하려면:

```bash
.venv/bin/catai-predict-cashlog path/to/product.jpg \
  --hybrid-config configs/cashlog/hybrid.serving.json \
  --device mps \
  --pretty
```

## FastAPI 서버

모델 추론까지 사용할 로컬 전용 서버:

```bash
export CATAI_CASHLOG_HYBRID_CONFIG=configs/cashlog/hybrid.serving.json
export CATAI_REQUIRE_INTERNAL_API_KEY=true
export CATAI_INTERNAL_API_KEY='<backend-only-secret>'
.venv/bin/catai-serve-cashlog --host 127.0.0.1 --port 8010
```

모바일 앱은 이 워커를 직접 호출하지 않습니다. 운영 요청은
`React Native -> Cloudflare -> Nginx -> NestJS -> Tailscale -> model worker`
경로만 허용하며 와일드카드 CORS와 공인 포트 개방은 사용하지 않습니다.

```bash
CATAI_CORS_ALLOWED_ORIGINS=https://api.your-cashlog-domain.example \
  catai-serve-cashlog --host 127.0.0.1 --port 8010
```

FastAPI 앱 entrypoint:

```text
main:app
```

Health check:

```bash
curl http://127.0.0.1:8010/health
```

JSON 분석 요청:

```bash
python - <<'PY'
import base64, json, urllib.request

image_base64 = base64.b64encode(open("path/to/product.jpg", "rb").read()).decode()
req = urllib.request.Request(
    "http://127.0.0.1:8010/analyze-image",
    data=json.dumps({"imageBase64": image_base64, "mimeType": "image/jpeg"}).encode(),
    headers={
        "Content-Type": "application/json",
        "X-Internal-API-Key": "<backend-only-secret>",
    },
)
print(urllib.request.urlopen(req).read().decode())
PY
```

Multipart 분석 요청:

```bash
curl \
  --header 'X-Internal-API-Key: <backend-only-secret>' \
  --form image=@path/to/product.jpg \
  http://127.0.0.1:8010/analyze-image
```

라벨링 리포트:

```text
http://127.0.0.1:8010/
http://127.0.0.1:8010/report
http://127.0.0.1:8010/report.json
```

FastAPI 엔드포인트:

| Method | Path | 설명 |
| --- | --- | --- |
| `GET` | `/` | 라벨링 가능한 모델 검수 리포트 HTML |
| `GET` | `/report` | 리포트 HTML 별칭 |
| `GET` | `/report.json` | 리포트 원본 JSON |
| `GET` | `/health` | report/checkpoint/model runtime 존재 여부 |
| `POST` | `/analyze-image` | 영수증·상품 이미지 33-leaf 추론. `[hybrid]` extra 필요 |

## Vercel 제한

기존 Vercel 설정은 정적 모델 리포트 확인용으로만 취급합니다.

- Vercel entrypoint: `main:app`
- `vercel.json`은 모든 요청을 `/main.py`로 rewrite합니다.
- `/`와 `/report`는 라벨링 리포트를 FastAPI가 서빙합니다.
- `/analyze-image`는 FastAPI 라우트로 존재하지만 하이브리드 모델 파일과
  비공개 네트워크 경계가 없는 서버리스 배포는 운영 추론 대상으로 삼지 않습니다.
- `pyproject.toml`의 `[tool.vercel]`도 `main:app`을 가리킵니다.

배포 제한: PyTorch/Transformers까지 Vercel Function에 넣으면 번들이
500MB 제한을 초과합니다. 그래서 Vercel은 라벨링 리포트 서버로 쓰고, 실제
모델 추론은 로컬 Mac, GPU 서버, 또는 별도 ML 서빙 환경에서
`pip install ".[hybrid]"`로 실행합니다.

Vercel이 자동 배포하면 아래 경로를 확인합니다.

```text
https://<deployment-url>/
https://<deployment-url>/health
https://<deployment-url>/report
```

실제 추론은 `docker-compose.prod.yml`의 loopback/Tailscale 전용 워커로
실행하고 NestJS 백엔드만 접근시킵니다.

## Cashlog 연결

현재 서빙 모델은 33개 leaf 전체를 반환하는
`cashlog33-hybrid-v1`입니다. `SigLIP2 + 한국어 RapidOCR + 텍스트 SGD +
CashLog 사전`을 결합하며, 실사진 holdout을 통과하기 전까지 모든 결과는
Top 3 사용자 확인 대상으로 반환합니다.

`/Users/uichan/workspace/cashlog`의 **서버 전용 환경 변수**에 아래 값을
설정하면 NestJS/API 계층이 비공개 모델 워커를 호출합니다. React Native 앱에
내부 URL이나 키를 넣지 않습니다.

```env
CATAI_PRODUCT_API_URL=http://127.0.0.1:8010/analyze-image
PRODUCT_ANALYZER_API_KEY=<backend-only-secret>
```

모델 서버에는 같은 값을 `CATAI_INTERNAL_API_KEY`로 설정합니다. CashLog
백엔드는 기존 `X-API-Key` 헤더를 서버 측에서 전달하며, 모델 워커는
`X-Internal-API-Key`와 이 호환 헤더를 모두 constant-time 비교합니다.

다른 호스트에 배치할 때는 Tailscale 주소를 사용하며, 모델 API를 Cloudflare
또는 공인 인터넷에 직접 공개하지 않습니다. 정확한 I/O와 보안 경계는
`ml_docs/CASHLOG33_MODEL_DESIGN.md` 및 `ml_docs/CASHLOG33_OPERATIONS.md`에
정리되어 있습니다. 실제 실행 결과와 실패 이력은
`ml_docs/CASHLOG33_RUN_LOG.md`에 있습니다.

## 모델 검수 리포트

33-leaf 데이터 출처, proxy/synthetic 평가 범위, per-class 지표, confusion
matrix, 승격 게이트를 정적 HTML 리포트로 생성합니다.

```bash
.venv/bin/python scripts/generate_cashlog33_report.py
```

생성 파일:

- `reports/cashlog33/model_report/index.html`
- `reports/cashlog33/model_report/REPORT.md`
- `reports/cashlog33/model_report/report.json`

Vercel에 이 저장소를 배포하면 `/` 또는 `/report`에서 같은 리포트를 볼 수
있습니다. 리포트는 FastAPI에서 직접 서빙하고, `/analyze-image`도 같은
FastAPI 앱에서 제공합니다.

현재 `guarded_integration_candidate` 상태이며 실사진 33-leaf 고정 holdout이
production gate를 통과하기 전에는 자동 확정을 활성화하지 않습니다.
