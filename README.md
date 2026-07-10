# Catai

Cashlog 상품 사진 카테고리 추천용 이미지 모델 패키지입니다.

현재 포함된 학습 결과:

- Model: `mobilenetv4_conv_small`
- Checkpoint: `checkpoints/cashlog_category_uecfood_mps/best.pt`
- 검증 성능: Top-1 `93.61%`, Top-3 `100.00%`
- 학습 범위: UECFood256에서 매핑 가능한 `식비`, `카페/간식`

## 설치

```bash
cd /Users/uichan/workspace/catai
.venv/bin/python -m pip install -e ".[serve]"
```

## CLI 예측

```bash
.venv/bin/catai-predict-cashlog path/to/product.jpg --device auto --pretty
```

Apple Silicon에서 MPS를 강제하려면:

```bash
.venv/bin/catai-predict-cashlog path/to/product.jpg --device mps --pretty
```

## API 서버

```bash
CATAI_DEVICE=mps .venv/bin/catai-serve-cashlog --host 127.0.0.1 --port 8010
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
있습니다. `vercel.json`은 루트 요청을
`reports/cashlog_model_report` 정적 output으로 배포합니다. Python 추론 서버는
로컬 개발용이고, Vercel 배포 대상은 라벨링 가능한 정적 리포트입니다.

MPS 접근이 제한된 환경에서는 `--device cpu`로 실행하면 됩니다.

나머지 Cashlog 카테고리는 아직 supervised 이미지 학습 데이터가 부족하므로,
추가 상품 데이터셋이 들어오기 전까지는 VLM fallback 또는 규칙 기반 매핑이
필요합니다.
