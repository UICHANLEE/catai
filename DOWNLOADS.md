# Dataset and Model Downloads

This workspace is organized for the pipeline:

| Stage | Target | Local path | Status |
| --- | --- | --- | --- |
| 1 | YOLO11n-seg | `models/detection/yolo11n-seg.pt` | public direct download |
| 1 | RPC | `data/raw/detection/rpc/` | Kaggle account/API token required |
| 1 | SKU110K | `data/raw/detection/sku110k/` | public direct download, academic/non-commercial terms |
| 1 | AI Hub product data | `data/raw/detection/aihub/` | AI Hub login and agreement required |
| 2 | MobileNetV4 | `models/classification/` | public Hugging Face/timm download |
| 2 | Products-10K | `data/raw/classification/products10k/` | Kaggle competition access required |
| 2 | ABO | `data/raw/classification/abo/` | public direct download |
| 2 | DeepFashion2 | `data/raw/classification/deepfashion2/` | Google Form and unzip password required |
| 2 | Food-101 | `data/raw/classification/food101/` | public direct download |
| 2 | UECFood256 | `data/raw/classification/uecfood256/` | public direct download |
| 2 | AI Hub product data | `data/raw/classification/aihub/` | AI Hub login and agreement required |
| 3 | OCR | Apple Vision OCR or PaddleOCR | no training dataset required |
| 4 | Category mapping | rules + LLM assist | no training dataset required |

Run downloads:

```bash
python3 scripts/download_assets.py models
python3 scripts/download_assets.py public-datasets
python3 scripts/download_assets.py best-effort
python3 scripts/download_assets.py asset --slug food101
```

Current local status:

| Asset | Path | Status |
| --- | --- | --- |
| YOLO11n-seg | `models/detection/yolo11n-seg.pt` | downloaded |
| MobileNetV4 Conv Small | `models/classification/mobilenetv4_conv_small.e2400_r224_in1k.safetensors` | downloaded |
| MobileNetV4 config | `models/classification/mobilenetv4_conv_small.e2400_r224_in1k.config.json` | downloaded |
| ABO listings | `data/raw/classification/abo/abo-listings.tar` | downloaded and tar-readable |
| UECFood256 | `data/raw/classification/uecfood256/dataset256.zip` | downloaded and zip-readable |
| Food-101 | `data/raw/classification/food101/food-101.tar.gz` | partial download; resume with slug `food101` |
| ABO images small | `data/raw/classification/abo/abo-images-small.tar` | not downloaded yet; use slug `abo-images-small` |
| SKU110K | `data/raw/detection/sku110k/SKU110K_fixed.tar.gz` | not downloaded yet; use slug `sku110k` |

Manual-gated datasets:

- RPC: download through Kaggle dataset "The Retail Product Checkout dataset" after accepting terms.
- Products-10K: download through the Kaggle competition page after account access is enabled.
- DeepFashion2: fill in the official Google Form to obtain the unzip password before downloading from Google Drive.
- AI Hub: log in to AI Hub, accept the dataset agreement, then place archives under the `aihub` directories above.

The large public archives are intentionally kept under `data/raw/...` and not unpacked automatically.
