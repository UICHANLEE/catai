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

나머지 Cashlog 카테고리는 아직 supervised 이미지 학습 데이터가 부족하므로,
추가 상품 데이터셋이 들어오기 전까지는 VLM fallback 또는 규칙 기반 매핑이
필요합니다.
