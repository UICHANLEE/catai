# Training

Current runnable training target:

- Model: `mobilenetv4_conv_small`
- Starting weights: `models/classification/mobilenetv4_conv_small.e2400_r224_in1k.safetensors`
- Dataset: `data/processed/classification/uecfood256/UECFOOD256`
- Classes: 256, mapped from `category.txt`
- Device for active document-aligned run: Apple MPS (`--device mps`, launched
  outside the sandbox so PyTorch can access Metal)

Completed run:

```bash
.venv/bin/python scripts/train_uecfood256_mobilenetv4.py \
  --epochs 1 \
  --batch-size 32 \
  --max-samples-per-class 10 \
  --log-interval 20
```

Result:

| Split | Loss | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| train | 5.478942 | 2.4414 | 7.3730 |
| val | 5.045927 | 7.6172 | 21.6797 |

Artifacts:

- `checkpoints/uecfood256_mobilenetv4/best.pt`
- `checkpoints/uecfood256_mobilenetv4/last.pt`
- `checkpoints/uecfood256_mobilenetv4/labels.json`
- `checkpoints/uecfood256_mobilenetv4/metrics.csv`

Full UECFood256 training command:

```bash
.venv/bin/python scripts/train_uecfood256_mobilenetv4.py \
  --epochs 5 \
  --batch-size 32 \
  --log-interval 50
```

Target 90% run:

```bash
bash scripts/train_until_90_uecfood256.sh
```

This uses bbox crop, RandAugment, random erasing, mixup, label smoothing,
one frozen-backbone warmup epoch, and stops early when validation top-1 reaches
90%.

Current long-run target:

```bash
bash scripts/train_until_90_plateau_uecfood256.sh
```

This uses all UECFood256 images, bbox crop, RandAugment, random erasing,
mixup, and label smoothing. It keeps training until validation top-1 reaches
90% and then stops only after 100 epochs without a further improvement of at
least 0.01 percentage points.

The long run has been launched detached:

```bash
.venv/bin/python scripts/launch_training.py uecfood256_target90_plateau100
```

Status files:

- Log: `logs/uecfood256_target90_plateau100.log`
- PID: `logs/uecfood256_target90_plateau100.pid`
- Output: `checkpoints/uecfood256_mobilenetv4_target90_plateau100/`

Check progress:

```bash
tail -f logs/uecfood256_target90_plateau100.log
cat checkpoints/uecfood256_mobilenetv4_target90_plateau100/metrics.csv
```

Monitor progress/restarts:

```bash
tail -f logs/uecfood256_target90_plateau100.monitor.log
```

In this CPU-only session, the small run processed 2,560 images in 558.5 seconds.
The full dataset has 31,910 images, so one full epoch is expected to take roughly
1.8 to 2.0 hours on the same runtime. If MPS becomes available in a normal local
terminal, use `--device mps`.

Detection/segmentation training is not started yet because the downloaded assets
do not currently include a usable detection dataset. `ABO listings` contains
metadata only, and `UECFood256` is suitable for classification.

## Cashlog B-plan alignment

The attached Cashlog B-plan document defines the MVP success metric at the
expense-category level:

- Top-1 Cashlog category accuracy: 70%+
- Top-3 Cashlog category accuracy: 90%+
- MVP pipeline: YOLO/YOLO11n detection -> crop classification or CLIP/VLM ->
  rule-based category mapper

The previous long run targeted 90% top-1 on 256 UECFood fine-grained food
classes. That is not the document metric, so it was stopped.

Current document-aligned local training:

```bash
.venv/bin/python scripts/train_cashlog_category_from_uecfood.py \
  --output-dir checkpoints/cashlog_category_uecfood_mobilenetv4 \
  --epochs 20 \
  --batch-size 32 \
  --log-interval 50
```

Current MPS-accelerated full run:

```bash
.venv/bin/python scripts/launch_training.py cashlog_category_uecfood_mps
```

Status files:

- Log: `logs/cashlog_category_uecfood_mps.log`
- PID: `logs/cashlog_category_uecfood_mps.pid`
- Output: `checkpoints/cashlog_category_uecfood_mps/`

This job uses all currently mapped UECFood256 images, bbox crop, RandAugment,
weighted sampling, class-weighted cross entropy, MobileNetV4 ImageNet weights,
30 epochs, batch size 32, and explicit `--device mps`.

Important limitation: with the datasets currently downloaded, image training
coverage is limited to food-like categories from UECFood256. The script maps
UECFood256 into the Cashlog categories it can honestly support:

- `식비`
- `카페/간식`

The remaining Cashlog categories (`생활용품`, `의류/패션`, `전자기기`,
`문화/교육`, `교통`, `의료/건강`, `미용`, `취미/여가`, `선물`, `기타`) require
Products-10K, ABO images, RPC, AI Hub product images, or user-collected product
photos before supervised category training is meaningful.

Smoke result for the aligned category script:

```text
train top1=83.56 top3=100.00
val top1=87.01 top3=100.00
```

This smoke run uses a tiny subset and only two covered categories, so it proves
the corrected training target and metric path, not full product-category
readiness.
