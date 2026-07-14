# Catai

Cashlog 상품 사진 카테고리 추천용 이미지 모델 패키지입니다.

현재 포함된 학습 결과:

- Model: `mobilenetv4_conv_small`
- Packaged checkpoint: `src/catai/assets/cashlog_category_uecfood_mps/best.pt`
- 검증 성능: Top-1 `93.61%`, Top-3 `100.00%`
- 학습 범위: UECFood256에서 매핑 가능한 `식비`, `카페/간식`

## 설치

```bash
cd /Users/uichan/workspace/catai
.venv/bin/python -m pip install -e .
```

모델 추론까지 로컬에서 실행하려면 PyTorch 계열 의존성을 포함해 설치합니다.

```bash
.venv/bin/python -m pip install -e ".[model]"
```

새 환경에서는 일반 pip로도 설치할 수 있습니다. Vercel 같은 서버리스 배포는
기본 설치만 사용하고, 모델 추론 서버는 별도 런타임에서 `[model]` extra로
실행합니다.

```bash
python -m pip install git+https://github.com/UICHANLEE/catai.git
python -m pip install "catai[model] @ git+https://github.com/UICHANLEE/catai.git"
```

패키지 안에 현재 best checkpoint가 포함되어 있어, 별도 모델 파일을 지정하지
않아도 `[model]` extra 설치 환경에서는 추론 서버가 동작합니다. 다른 checkpoint를 쓰려면
`CATAI_CASHLOG_CHECKPOINT` 환경 변수로 경로를 지정합니다.

## CLI 예측

```bash
.venv/bin/catai-predict-cashlog path/to/product.jpg --device auto --pretty
```

Apple Silicon에서 MPS를 강제하려면:

```bash
.venv/bin/catai-predict-cashlog path/to/product.jpg --device mps --pretty
```

## FastAPI 서버

모델 추론까지 사용할 로컬 서버:

```bash
CATAI_DEVICE=mps .venv/bin/catai-serve-cashlog --host 127.0.0.1 --port 8010
```

배포 환경에서는 Cashlog 앱 주소만 CORS 허용 목록에 등록합니다. 와일드카드(`*`)는 사용하지 않습니다.

```bash
CATAI_CORS_ALLOWED_ORIGINS=https://your-cashlog-domain.example \
  catai-serve-cashlog --host 0.0.0.0 --port 8010
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
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode())
PY
```

Multipart 분석 요청:

```bash
curl -F image=@path/to/product.jpg http://127.0.0.1:8010/analyze-image
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
| `POST` | `/analyze-image` | 상품 이미지 카테고리 추론. `[model]` extra 필요 |

## Vercel 배포

이 저장소는 FastAPI 앱으로 배포됩니다.

- Vercel entrypoint: `main:app`
- `vercel.json`은 모든 요청을 `/main.py`로 rewrite합니다.
- `/`와 `/report`는 라벨링 리포트를 FastAPI가 서빙합니다.
- `/analyze-image`는 FastAPI 라우트로 존재하지만, Vercel 서버리스 번들 크기
  제한 때문에 PyTorch 모델 런타임은 포함하지 않습니다. Vercel에서는 모델
  런타임이 없으면 `503`을 반환합니다.
- `pyproject.toml`의 `[tool.vercel]`도 `main:app`을 가리킵니다.

배포 제한: PyTorch/torchvision/timm까지 Vercel Function에 넣으면 번들이
500MB 제한을 초과합니다. 그래서 Vercel은 라벨링 리포트 서버로 쓰고, 실제
모델 추론은 로컬 Mac, GPU 서버, 또는 별도 ML 서빙 환경에서
`pip install ".[model]"`로 실행합니다.

Vercel이 자동 배포하면 아래 경로를 확인합니다.

```text
https://<deployment-url>/
https://<deployment-url>/health
https://<deployment-url>/report
```

모델 추론까지 Vercel 배포에서 시도하려면 `requirements.txt`가 `.[model]`
extra를 설치합니다. 배포 후 `/health`에서 `checkpoint_available: true`를
확인하고, `/analyze-image`가 `model dependencies are not installed`를
반환하지 않아야 합니다.

주의: PyTorch/torchvision/timm 조합은 Vercel Function 용량 제한에 걸릴 수
있습니다. 이 경우 Vercel은 리포트/헬스체크 용도로 두고, 실제 모델 API는
Render, Railway, Fly.io, Hugging Face Spaces, Modal 같은 모델 서버 런타임에
배포한 뒤 Cashlog의 `PRODUCT_ANALYZER_API_URL`에 그 URL을 넣습니다.

## Cashlog 연결

`/Users/uichan/workspace/cashlog`에서 서버 환경 변수에 아래 값을 설정하면
`api/analyze-image.ts`가 OpenAI/VLM fallback 전에 이 로컬 모델 서버를 먼저
호출합니다.

```env
CATAI_PRODUCT_API_URL=http://127.0.0.1:8010/analyze-image
```

반환 카테고리 매핑:

- `식비` -> `meal_dining`
- `카페/간식` -> `meal_cafe`

## 모델 검수 리포트

학습 데이터 이미지, 원본 UECFood 라벨, Cashlog 학습 라벨, 모델 추론 결과를
나란히 비교하고, 브라우저에서 직접 수정 라벨을 붙일 수 있는 정적 HTML
리포트를 생성합니다.

```bash
.venv/bin/catai-report-cashlog \
  --split val \
  --limit 120 \
  --device mps \
  --output-dir reports/cashlog_model_report
```

생성 파일:

- `reports/cashlog_model_report/index.html`: 브라우저로 여는 시각 검수 리포트
- `reports/cashlog_model_report/report.json`: 같은 내용을 담은 JSON 결과

리포트에서 할 수 있는 일:

- 모델 추론 결과와 학습 라벨 비교
- Cashlog 라벨 직접 선택
- 모델 결과 채택
- 검수 메모 입력
- 브라우저 `localStorage` 자동 저장
- 라벨링 결과를 JSON/CSV로 내보내기

Vercel에 이 저장소를 배포하면 `/` 또는 `/report`에서 같은 리포트를 볼 수
있습니다. 리포트는 FastAPI에서 직접 서빙하고, `/analyze-image`도 같은
FastAPI 앱에서 제공합니다.

MPS 접근이 제한된 환경에서는 `--device cpu`로 실행하면 됩니다.

나머지 Cashlog 카테고리는 아직 supervised 이미지 학습 데이터가 부족하므로,
추가 상품 데이터셋이 들어오기 전까지는 VLM fallback 또는 규칙 기반 매핑이
필요합니다.
