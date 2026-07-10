from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .cashlog_classifier import CASHLOG_LEAF_BY_MODEL_ID, CashlogCategoryClassifier, crop_bbox


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PACKAGE_ROOT / "data/processed/classification/uecfood256/UECFOOD256"
DEFAULT_OVERRIDES = PACKAGE_ROOT / "configs/cashlog/uecfood_category_overrides.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "reports/cashlog_model_report"
DEFAULT_METRICS = PACKAGE_ROOT / "checkpoints/cashlog_category_uecfood_mps/metrics.csv"

CASHLOG_LABEL_CHOICES = [
    ("meal_grocery", "식료품"),
    ("meal_dining", "식비"),
    ("meal_cafe", "카페/간식"),
    ("meal_drink", "음료"),
    ("life_goods", "생활용품"),
    ("life_appliance", "전자기기/가전"),
    ("life_clean", "청소/위생"),
    ("fashion_clothes", "의류/패션"),
    ("fashion_beauty", "미용"),
    ("health_med", "의료/건강"),
    ("edu_book", "도서/교육"),
    ("leisure_hobby", "취미/여가"),
    ("gift_present", "선물"),
    ("misc_uncat", "미분류"),
    ("misc_other", "기타"),
]


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    source_label: str
    true_model_id: str
    true_display_name: str
    true_cashlog_leaf_id: str
    bbox: tuple[int, int, int, int] | None


def infer_cashlog_category(source_label: str, overrides: dict[str, list[str]]) -> str:
    lowered = source_label.lower()
    for keyword in overrides["cafe_snack_keywords"]:
        if keyword in lowered:
            return "cafe_snack"
    for keyword in overrides["food_keywords"]:
        if keyword in lowered:
            return "food"
    return "food"


def read_uec_categories(dataset_root: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    with (dataset_root / "category.txt").open() as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            raw_id, name = line.rstrip("\n").split("\t", 1)
            rows[int(raw_id)] = name
    return rows


def read_bboxes(class_dir: Path) -> dict[str, tuple[int, int, int, int]]:
    bbox_path = class_dir / "bb_info.txt"
    if not bbox_path.exists():
        return {}
    boxes: dict[str, tuple[int, int, int, int]] = {}
    with bbox_path.open() as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            image_id, x1, y1, x2, y2 = parts
            boxes[image_id] = (int(x1), int(y1), int(x2), int(y2))
    return boxes


def collect_samples(dataset_root: Path, classifier: CashlogCategoryClassifier, overrides_path: Path) -> list[DatasetSample]:
    overrides = json.loads(overrides_path.read_text())
    labels_by_id = {str(row["id"]): str(row.get("display_name") or row["id"]) for row in classifier.categories}
    samples: list[DatasetSample] = []
    for class_id, source_label in sorted(read_uec_categories(dataset_root).items()):
        model_id = infer_cashlog_category(source_label, overrides)
        if model_id not in labels_by_id:
            continue
        class_dir = dataset_root / str(class_id)
        boxes = read_bboxes(class_dir)
        for path in sorted(class_dir.glob("*.jpg")):
            samples.append(
                DatasetSample(
                    path=path,
                    source_label=source_label,
                    true_model_id=model_id,
                    true_display_name=labels_by_id[model_id],
                    true_cashlog_leaf_id=CASHLOG_LEAF_BY_MODEL_ID.get(model_id, "misc_uncat"),
                    bbox=boxes.get(path.stem),
                )
            )
    return samples


def stratified_split(
    samples: list[DatasetSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    by_label: dict[str, list[DatasetSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.true_model_id, []).append(sample)
    rng = random.Random(seed)
    train: list[DatasetSample] = []
    val: list[DatasetSample] = []
    for label_samples in by_label.values():
        shuffled = label_samples[:]
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def balanced_subset(samples: list[DatasetSample], limit: int | None, seed: int) -> list[DatasetSample]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    if limit is None or limit >= len(shuffled):
        return shuffled

    by_label: dict[str, list[DatasetSample]] = {}
    for sample in shuffled:
        by_label.setdefault(sample.true_model_id, []).append(sample)

    selected: list[DatasetSample] = []
    while len(selected) < limit and any(by_label.values()):
        for label in sorted(by_label):
            if by_label[label] and len(selected) < limit:
                selected.append(by_label[label].pop())
    rng.shuffle(selected)
    return selected


def image_data_uri(image: Image.Image, max_side: int = 320) -> str:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def sample_images(sample: DatasetSample, use_bbox: bool) -> tuple[str, str | None]:
    with Image.open(sample.path) as raw:
        original = raw.convert("RGB")
        original_uri = image_data_uri(original)
        if not use_bbox or sample.bbox is None:
            return original_uri, None
        cropped = crop_bbox(original, sample.bbox)
        return original_uri, image_data_uri(cropped)


def read_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    rows = list(csv.DictReader(metrics_path.open()))
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row["val_top1"]))
    last = rows[-1]
    return {
        "best_epoch": int(best["epoch"]),
        "best_val_top1": float(best["val_top1"]),
        "best_val_top3": float(best["val_top3"]),
        "last_epoch": int(last["epoch"]),
        "last_val_top1": float(last["val_top1"]),
        "last_val_top3": float(last["val_top3"]),
    }


def pct(value: float) -> str:
    return f"{value:.2f}%"


def render_bar(confidence: float) -> str:
    width = max(0, min(100, confidence * 100))
    return (
        '<div class="bar">'
        f'<span style="width:{width:.2f}%"></span>'
        f'<b>{confidence * 100:.1f}%</b>'
        "</div>"
    )


def render_label_options() -> str:
    options = ['<option value="">라벨 선택</option>']
    options.extend(
        f'<option value="{html.escape(value)}">{html.escape(label)} ({html.escape(value)})</option>'
        for value, label in CASHLOG_LABEL_CHOICES
    )
    return "\n".join(options)


def labeling_script() -> str:
    return """
  <script>
    const STORAGE_KEY = "cashlog-model-report-labels-v1";
    const cards = Array.from(document.querySelectorAll("article[data-sample-id]"));
    const state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    const filterSelect = document.querySelector("#filter");
    const progress = document.querySelector("#label-progress");

    function saveState() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      updateProgress();
    }

    function recordFor(card) {
      const id = card.dataset.sampleId;
      if (!state[id]) {
        state[id] = { userLabel: "", note: "", updatedAt: "" };
      }
      return state[id];
    }

    function applyState(card) {
      const record = recordFor(card);
      const select = card.querySelector(".label-select");
      const note = card.querySelector(".note-input");
      if (select) select.value = record.userLabel || "";
      if (note) note.value = record.note || "";
      card.dataset.userLabel = record.userLabel || "";
      card.classList.toggle("labeled", Boolean(record.userLabel));
    }

    function updateProgress() {
      const labeled = cards.filter((card) => Boolean(recordFor(card).userLabel)).length;
      progress.textContent = `${labeled}/${cards.length} 라벨 완료`;
    }

    function applyFilter() {
      const mode = filterSelect.value;
      cards.forEach((card) => {
        const isWrong = card.dataset.correct !== "true";
        const isLabeled = Boolean(recordFor(card).userLabel);
        const show =
          mode === "all" ||
          (mode === "wrong" && isWrong) ||
          (mode === "unlabeled" && !isLabeled) ||
          (mode === "labeled" && isLabeled);
        card.hidden = !show;
      });
    }

    function rowsForExport() {
      return cards.map((card) => {
        const record = recordFor(card);
        return {
          sample_id: card.dataset.sampleId,
          image_path: card.dataset.path,
          source_label: card.dataset.sourceLabel,
          training_label: card.dataset.trueCategory,
          training_cashlog_leaf_id: card.dataset.trueLeaf,
          predicted_label: card.dataset.predictedCategory,
          predicted_cashlog_leaf_id: card.dataset.predictedLeaf,
          confidence: Number(card.dataset.confidence || 0),
          model_correct: card.dataset.correct === "true",
          user_label: record.userLabel || "",
          user_note: record.note || "",
          updated_at: record.updatedAt || "",
        };
      });
    }

    function download(filename, type, content) {
      const blob = new Blob([content], { type });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    function csvEscape(value) {
      const text = String(value ?? "");
      return `"${text.replaceAll('"', '""')}"`;
    }

    function exportJson() {
      download(
        "cashlog_model_labels.json",
        "application/json",
        JSON.stringify({ exported_at: new Date().toISOString(), rows: rowsForExport() }, null, 2),
      );
    }

    function exportCsv() {
      const rows = rowsForExport();
      const headers = Object.keys(rows[0] || {});
      const csv = [
        headers.join(","),
        ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
      ].join("\\n");
      download("cashlog_model_labels.csv", "text/csv;charset=utf-8", csv);
    }

    cards.forEach((card) => {
      applyState(card);
      const select = card.querySelector(".label-select");
      const note = card.querySelector(".note-input");
      const accept = card.querySelector(".accept-btn");

      select?.addEventListener("change", () => {
        const record = recordFor(card);
        record.userLabel = select.value;
        record.updatedAt = new Date().toISOString();
        card.dataset.userLabel = record.userLabel;
        card.classList.toggle("labeled", Boolean(record.userLabel));
        saveState();
        applyFilter();
      });

      note?.addEventListener("input", () => {
        const record = recordFor(card);
        record.note = note.value;
        record.updatedAt = new Date().toISOString();
        saveState();
      });

      accept?.addEventListener("click", () => {
        const record = recordFor(card);
        record.userLabel = card.dataset.predictedLeaf || "";
        record.updatedAt = new Date().toISOString();
        applyState(card);
        saveState();
        applyFilter();
      });
    });

    filterSelect?.addEventListener("change", applyFilter);
    document.querySelector("#export-json")?.addEventListener("click", exportJson);
    document.querySelector("#export-csv")?.addEventListener("click", exportCsv);
    document.querySelector("#clear-labels")?.addEventListener("click", () => {
      if (!confirm("브라우저에 저장된 라벨을 모두 지울까요?")) return;
      localStorage.removeItem(STORAGE_KEY);
      Object.keys(state).forEach((key) => delete state[key]);
      cards.forEach(applyState);
      saveState();
      applyFilter();
    });

    updateProgress();
    applyFilter();
  </script>
"""


def render_html(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    metric = report["training_metrics"]
    cards = [
        ("학습 Best Top-1", pct(metric.get("best_val_top1", 0.0))),
        ("학습 Best Top-3", pct(metric.get("best_val_top3", 0.0))),
        ("리포트 샘플 Top-1", pct(report["sample_accuracy_top1"])),
        ("검수 이미지", f'{report["sample_count"]}장'),
    ]
    card_html = "\n".join(
        f'<section class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>'
        for label, value in cards
    )
    row_html = "\n".join(render_row(row) for row in rows)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cashlog 모델 검수 리포트</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #202124;
      --muted: #687078;
      --line: #d7dce2;
      --ok: #197a47;
      --bad: #b3261e;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --accent: #1267c4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    .sub {{ color: var(--muted); margin: 0; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      padding: 20px 32px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 24px; }}
    .note {{
      margin: 0 32px 20px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: #fff8e6;
      border-radius: 8px;
      color: #5b4300;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin: 0 32px 20px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, .94);
      backdrop-filter: blur(10px);
    }}
    .toolbar select,
    .toolbar button,
    .label-panel select,
    .label-panel input {{
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    .toolbar select,
    .label-panel select,
    .label-panel input {{ padding: 0 10px; }}
    .toolbar button,
    .label-panel button {{
      min-height: 36px;
      border: 1px solid #9db7d3;
      border-radius: 6px;
      background: #eef6ff;
      color: #064b86;
      font: inherit;
      cursor: pointer;
      padding: 0 12px;
    }}
    .toolbar .danger {{
      border-color: #e7b0aa;
      background: #fff2f0;
      color: var(--bad);
    }}
    .progress {{ margin-left: auto; color: var(--muted); font-size: 13px; }}
    main {{
      display: grid;
      gap: 14px;
      padding: 0 32px 40px;
    }}
    article {{
      display: grid;
      grid-template-columns: 180px 180px minmax(220px, 1fr) minmax(220px, 1fr);
      gap: 16px;
      align-items: stretch;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .image-box {{
      min-height: 152px;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: grid;
      place-items: center;
      overflow: hidden;
      background: #f1f3f5;
    }}
    .image-box img {{ max-width: 100%; max-height: 172px; display: block; }}
    .caption {{ margin-top: 8px; color: var(--muted); font-size: 12px; word-break: break-all; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
    .value {{ margin: 3px 0 12px; font-size: 17px; font-weight: 700; }}
    .source {{ color: var(--muted); font-size: 13px; }}
    .ok {{ color: var(--ok); }}
    .bad {{ color: var(--bad); }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 13px;
      border: 1px solid currentColor;
    }}
    .bar {{
      position: relative;
      height: 24px;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      margin: 8px 0;
      background: #f7f8fa;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: #b9dcff;
    }}
    .bar b {{
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      font-size: 12px;
    }}
    .top-list {{ margin: 8px 0 0; padding: 0; list-style: none; }}
    .top-list li {{ margin: 5px 0; color: var(--muted); }}
    .label-panel {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: minmax(220px, 320px) auto minmax(220px, 1fr);
      gap: 10px;
      align-items: center;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }}
    .label-panel input {{ width: 100%; }}
    article.labeled {{
      outline: 2px solid #9bd2b4;
      outline-offset: 0;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    @media (max-width: 980px) {{
      article {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 620px) {{
      header, .metrics, main {{ padding-left: 16px; padding-right: 16px; }}
      .note, .toolbar {{ margin-left: 16px; margin-right: 16px; }}
      article {{ grid-template-columns: 1fr; }}
      .label-panel {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Cashlog 모델 검수 리포트</h1>
    <p class="sub">학습 데이터 이미지, 라벨링, 모델 추론 결과를 한 화면에서 비교합니다.</p>
  </header>
  <section class="metrics">{card_html}</section>
  <p class="note">현재 checkpoint는 UECFood256으로부터 매핑 가능한 <b>식비</b>, <b>카페/간식</b> 두 범위만 supervised 학습했습니다. 전체 Cashlog 12개 카테고리 성능 리포트로 해석하면 안 됩니다.</p>
  <section class="toolbar" aria-label="라벨링 도구">
    <select id="filter" aria-label="샘플 필터">
      <option value="all">전체 보기</option>
      <option value="wrong">모델 불일치만</option>
      <option value="unlabeled">미라벨만</option>
      <option value="labeled">라벨 완료만</option>
    </select>
    <button id="export-json" type="button">JSON 내보내기</button>
    <button id="export-csv" type="button">CSV 내보내기</button>
    <button id="clear-labels" class="danger" type="button">라벨 초기화</button>
    <span id="label-progress" class="progress">0/{report["sample_count"]} 라벨 완료</span>
  </section>
  <main>{row_html}</main>
  {labeling_script()}
</body>
</html>
"""


def render_row(row: dict[str, Any]) -> str:
    status_class = "ok" if row["correct"] else "bad"
    status_text = "일치" if row["correct"] else "불일치"
    crop = (
        f'<div class="image-box"><img src="{row["crop_uri"]}" alt="model input crop"></div>'
        if row.get("crop_uri")
        else '<div class="image-box"><span class="source">bbox 없음<br>원본 사용</span></div>'
    )
    top_items = "".join(
        f'<li>{html.escape(item["display_name"])} / {html.escape(item["category"])} {item["confidence"] * 100:.1f}%</li>'
        for item in row["top_predictions"]
    )
    sample_id = html.escape(row["path"], quote=True)
    return f"""<article
  data-sample-id="{sample_id}"
  data-path="{sample_id}"
  data-source-label="{html.escape(row["source_label"], quote=True)}"
  data-true-category="{html.escape(row["true_category"], quote=True)}"
  data-true-leaf="{html.escape(row["true_cashlog_leaf_id"], quote=True)}"
  data-predicted-category="{html.escape(row["predicted_category"], quote=True)}"
  data-predicted-leaf="{html.escape(row["predicted_cashlog_leaf_id"], quote=True)}"
  data-confidence="{row["confidence"]:.8f}"
  data-correct="{str(row["correct"]).lower()}"
>
  <section>
    <div class="image-box"><img src="{row["original_uri"]}" alt="training sample"></div>
    <div class="caption"><code>{html.escape(row["path"])}</code></div>
  </section>
  <section>
    {crop}
    <div class="caption">모델 입력 이미지</div>
  </section>
  <section>
    <div class="label">학습 데이터 라벨</div>
    <div class="value">{html.escape(row["true_display_name"])} <span class="source">({html.escape(row["true_category"])})</span></div>
    <div class="label">UECFood 원본 라벨</div>
    <div class="value">{html.escape(row["source_label"])}</div>
    <div class="label">Cashlog leaf</div>
    <div class="value"><code>{html.escape(row["true_cashlog_leaf_id"])}</code></div>
  </section>
  <section>
    <span class="pill {status_class}">{status_text}</span>
    <div class="label" style="margin-top:12px">모델 추론 결과</div>
    <div class="value">{html.escape(row["predicted_display_name"])} <span class="source">({html.escape(row["predicted_category"])})</span></div>
    {render_bar(row["confidence"])}
    <ul class="top-list">{top_items}</ul>
  </section>
  <section class="label-panel" aria-label="라벨 수정">
    <select class="label-select" aria-label="수정 라벨">
      {render_label_options()}
    </select>
    <button class="accept-btn" type="button">모델 결과 채택</button>
    <input class="note-input" type="text" placeholder="검수 메모 또는 라벨링 근거">
  </section>
</article>"""


def build_report(args: argparse.Namespace) -> Path:
    classifier = CashlogCategoryClassifier(
        checkpoint_path=args.checkpoint,
        labels_path=args.labels,
        device=args.device,
    )
    samples = collect_samples(args.dataset_root, classifier, args.overrides)
    train_samples, val_samples = stratified_split(samples, args.val_ratio, args.seed)
    split_samples = {
        "train": train_samples,
        "val": val_samples,
        "all": samples,
    }[args.split]
    selected = balanced_subset(split_samples, args.limit, args.seed)

    rows: list[dict[str, Any]] = []
    correct_count = 0
    for sample in selected:
        bbox = sample.bbox if args.use_bbox else None
        predictions = classifier.predict(sample.path, top_k=3, bbox=bbox, bbox_padding=args.bbox_padding)
        best = predictions[0]
        correct = best.model_id == sample.true_model_id
        correct_count += int(correct)
        original_uri, crop_uri = sample_images(sample, args.use_bbox)
        rows.append(
            {
                "path": str(sample.path.relative_to(PACKAGE_ROOT)),
                "source_label": sample.source_label,
                "true_category": sample.true_model_id,
                "true_display_name": sample.true_display_name,
                "true_cashlog_leaf_id": sample.true_cashlog_leaf_id,
                "predicted_category": best.model_id,
                "predicted_display_name": best.display_name,
                "predicted_cashlog_leaf_id": best.cashlog_leaf_id,
                "confidence": best.confidence,
                "correct": correct,
                "bbox": sample.bbox,
                "original_uri": original_uri,
                "crop_uri": crop_uri,
                "top_predictions": [
                    {
                        "category": prediction.model_id,
                        "display_name": prediction.display_name,
                        "cashlog_leaf_id": prediction.cashlog_leaf_id,
                        "confidence": prediction.confidence,
                    }
                    for prediction in predictions
                ],
            }
        )

    sample_accuracy = correct_count * 100.0 / max(1, len(rows))
    report = {
        "checkpoint": str(classifier.checkpoint_path),
        "device": str(classifier.device),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "sample_count": len(rows),
        "sample_accuracy_top1": sample_accuracy,
        "use_bbox": args.use_bbox,
        "training_metrics": read_metrics(args.metrics),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps({"summary": report, "rows": rows}, ensure_ascii=False, indent=2)
    )
    (args.output_dir / "index.html").write_text(render_html(report, rows))
    return args.output_dir / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a visual Cashlog model inspection report.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bbox-padding", type=float, default=0.10)
    parser.add_argument("--no-bbox", dest="use_bbox", action="store_false")
    parser.set_defaults(use_bbox=True)
    args = parser.parse_args()

    output_path = build_report(args)
    print(output_path)


if __name__ == "__main__":
    main()
