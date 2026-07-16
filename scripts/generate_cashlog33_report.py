#!/usr/bin/env python3
"""Generate the durable CashLog 33 technical model report and visual evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/cashlog33/model_report"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def svg_bar_chart(
    path: Path,
    title: str,
    rows: list[tuple[str, float]],
    maximum: float | None = None,
    value_format: str = "number",
) -> None:
    width = 1120
    row_height = 27
    top = 76
    bottom = 45
    label_width = 260
    chart_width = width - label_width - 110
    height = top + bottom + row_height * len(rows)
    max_value = maximum or max((value for _, value in rows), default=1.0) or 1.0
    colors = ["#f97316", "#059669", "#2563eb", "#db2777", "#7c3aed"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="38" font-family="system-ui,sans-serif" font-size="22" font-weight="700" fill="#111827">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(rows):
        y = top + index * row_height
        bar_width = max(1.0, chart_width * min(value / max_value, 1.0))
        display = pct(value) if value_format == "percent" else f"{int(value):,}"
        parts.extend(
            [
                f'<text x="24" y="{y + 17}" font-family="ui-monospace,monospace" font-size="13" fill="#374151">{html.escape(label)}</text>',
                f'<rect x="{label_width}" y="{y + 4}" width="{chart_width}" height="16" rx="2" fill="#eef2f7"/>',
                f'<rect x="{label_width}" y="{y + 4}" width="{bar_width:.2f}" height="16" rx="2" fill="{colors[index % len(colors)]}"/>',
                f'<text x="{label_width + chart_width + 12}" y="{y + 17}" font-family="system-ui,sans-serif" font-size="13" fill="#111827">{display}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_candidate_chart(
    path: Path,
    vision: dict[str, Any],
    e2e: dict[str, Any],
    airflow_e2e: dict[str, Any],
) -> None:
    rows = [
        ("SigLIP2 zero-shot / Open Images proxy", vision["test_zero_shot_same_split"]["top1_accuracy"]),
        ("SigLIP2 linear head / same proxy", vision["test_linear_head"]["top1_accuracy"]),
        ("Serving hybrid / Mac runtime, fixed Noto fixtures", e2e["top1_accuracy"]),
        ("Airflow candidate / Linux runtime, same fixtures", airflow_e2e["top1_accuracy"]),
    ]
    svg_bar_chart(path, "Candidate Top-1 by evaluation scope", rows, maximum=1.0, value_format="percent")


def svg_confusion(path: Path, categories: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> None:
    ids = [str(row["id"]) for row in categories]
    index = {leaf_id: position for position, leaf_id in enumerate(ids)}
    matrix = [[0 for _ in ids] for _ in ids]
    for row in predictions:
        matrix[index[str(row["expected"])]] [index[str(row["predicted"])]] += 1
    max_value = max(max(row) for row in matrix) or 1
    cell = 22
    left = 230
    top = 230
    width = left + cell * len(ids) + 40
    height = top + cell * len(ids) + 50
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="system-ui,sans-serif" font-size="22" font-weight="700" fill="#111827">Synthetic E2E confusion matrix</text>',
        '<text x="24" y="58" font-family="system-ui,sans-serif" font-size="13" fill="#6b7280">Rows: expected, columns: predicted. This is an integration fixture, not real-photo accuracy.</text>',
    ]
    for position, leaf_id in enumerate(ids):
        y = top + position * cell + 16
        x = left + position * cell + 15
        parts.append(
            f'<text x="{left - 8}" y="{y}" text-anchor="end" font-family="ui-monospace,monospace" font-size="10" fill="#374151">{leaf_id}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{top - 8}" transform="rotate(-55 {x} {top - 8})" text-anchor="start" font-family="ui-monospace,monospace" font-size="10" fill="#374151">{leaf_id}</text>'
        )
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = value / max_value
            red = round(244 - 205 * intensity)
            green = round(247 - 91 * intensity)
            blue = round(250 - 126 * intensity)
            x = left + col_index * cell
            y = top + row_index * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="rgb({red},{green},{blue})"/>'
            )
            if value:
                parts.append(
                    f'<text x="{x + cell / 2:.1f}" y="{y + 15}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" font-weight="700" fill="{("#ffffff" if intensity > 0.55 else "#111827")}">{value}</text>'
                )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_table(categories: list[dict[str, Any]], text_quality: dict[str, Any]) -> str:
    lines = ["| Leaf ID | Group | Display name | Text rows |", "|---|---|---|---:|"]
    for row in categories:
        leaf_id = str(row["id"])
        lines.append(
            f"| `{leaf_id}` | {row['group_name']} | {row['display_name']} | "
            f"{text_quality['counts'][leaf_id]['total']:,} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = load_json(ROOT / "configs/cashlog/categories.json")
    selection = load_json(ROOT / "reports/cashlog33/model_selection.json")
    vision = load_json(ROOT / "checkpoints/cashlog33/vision_head_v1/metrics.json")
    text_metrics = load_json(ROOT / "checkpoints/cashlog33/text_sgd_v2/metrics.json")
    text_quality = load_json(ROOT / "data/processed/cashlog33/text/v2/quality_report.json")
    openimages = load_json(ROOT / "data/raw/cashlog33/openimages_v7/collection_summary.json")
    e2e = load_json(ROOT / "reports/cashlog33/hybrid_e2e/metrics.json")
    airflow_e2e = load_json(
        ROOT / "reports/cashlog33/airflow_latest/hybrid_e2e/metrics.json"
    )
    airflow_selection = load_json(
        ROOT / "reports/cashlog33/airflow_latest/model_selection.json"
    )
    per_class = read_csv(ROOT / "reports/cashlog33/hybrid_e2e/per_class_metrics.csv")
    predictions = load_jsonl(ROOT / "reports/cashlog33/hybrid_e2e/predictions.jsonl")
    config = load_json(ROOT / "configs/cashlog/hybrid.serving.json")

    output = args.output_dir.resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    class_count_chart = assets / "text-class-counts.svg"
    candidate_chart = assets / "candidate-top1.svg"
    per_class_chart = assets / "hybrid-per-class-f1.svg"
    confusion_chart = assets / "hybrid-confusion.svg"
    svg_bar_chart(
        class_count_chart,
        "Text training rows by CashLog leaf",
        [(str(row["id"]), float(text_quality["counts"][str(row["id"])]["total"])) for row in categories],
    )
    svg_candidate_chart(candidate_chart, vision, e2e, airflow_e2e)
    svg_bar_chart(
        per_class_chart,
        "Hybrid synthetic E2E F1 by leaf",
        [(row["leaf_id"], float(row["f1"])) for row in per_class],
        maximum=1.0,
        value_format="percent",
    )
    svg_confusion(confusion_chart, categories, predictions)

    generated_at = datetime.now(timezone.utc).isoformat()
    wrong = [row for row in predictions if row["expected"] != row["predicted"]]
    fallback_errors = Counter(str(row["predicted"]) for row in wrong)
    source = text_quality["source_metadata"]
    report_md = f"""# CashLog 33-Leaf Hybrid Model Report

Generated: `{generated_at}`
Taxonomy: `{config['taxonomy_version']}`
Candidate: `{selection['selected_candidate']}`
Decision: **{selection['decision']}**

## Technical Summary

The selected architecture is a guarded ensemble of frozen SigLIP2 image embeddings, a
CashLog-specific linear visual head, Korean RapidOCR, a word/character TF-IDF text classifier,
and an auditable merchant/keyword lexicon. It always returns three of the exact 33 leaf IDs.
Unreadable or semantically empty receipts are routed to `misc_uncat`; automatic confirmation is
disabled until a frozen real CashLog photo holdout passes all promotion gates.

This candidate is usable for **Top-3 recommendation integration**, but it is **not certified as a
production-accuracy model**. The missing evidence is a manually labeled real-photo holdout with at
least 10 photos per leaf (330 total).

## Architecture

```mermaid
flowchart LR
  A["JPEG/PNG/WebP/HEIF image"] --> B["API signature and size validation"]
  B --> C["SigLIP2 image embedding"]
  C --> D["Zero-shot prompts"]
  C --> E["CashLog linear vision head"]
  B --> F["RapidOCR Korean"]
  F --> G["Word + character TF-IDF SGD"]
  F --> H["Auditable OCR lexicon"]
  D --> I["Weighted fusion"]
  E --> I
  G --> I
  H --> I
  I --> J["Calibration and safe fallback"]
  J --> K["Top-3 + need_user_check"]
```

The visual head covers 23 visually grounded leaves. Ten document/open-set leaves are supplied by
OCR/text and decision logic. `misc_uncat` is a fallback decision, not a visual object class.

## Key Results

| Evidence | Samples | Leaves | Top-1 | Top-3 | Macro F1 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| SigLIP2 zero-shot, Open Images test split | {vision['test_zero_shot_same_split']['samples']} | 23 | {pct(vision['test_zero_shot_same_split']['top1_accuracy'])} | {pct(vision['test_zero_shot_same_split']['top3_accuracy'])} | {pct(vision['test_zero_shot_same_split']['macro_f1'])} | Visual proxy only |
| SigLIP2 linear head, same split | {vision['test_linear_head']['samples']} | 23 | {pct(vision['test_linear_head']['top1_accuracy'])} | {pct(vision['test_linear_head']['top3_accuracy'])} | {pct(vision['test_linear_head']['macro_f1'])} | Visual proxy only |
| OCR text classifier | {text_metrics['test']['samples']:,} | 33 | {pct(text_metrics['test']['top1_accuracy'])} | {pct(text_metrics['test']['top3_accuracy'])} | {pct(text_metrics['test']['macro_f1'])} | Synthetic/weak text proxy; inflated |
| Serving hybrid, Mac runtime on fixed Noto fixtures | {e2e['samples']} | 33 | {pct(e2e['top1_accuracy'])} | {pct(e2e['top3_accuracy'])} | {pct(e2e['macro_f1'])} | Synthetic I/O test only |
| Airflow candidate, Linux runtime on the same fixtures | {airflow_e2e['samples']} | 33 | {pct(airflow_e2e['top1_accuracy'])} | {pct(airflow_e2e['top3_accuracy'])} | {pct(airflow_e2e['macro_f1'])} | Synthetic reproducibility run only |

The hybrid synthetic run had false auto-confirm rate `{pct(e2e['false_auto_confirm_rate'])}` because
`allow_auto_confirm=false`. CPU latency was p50 `{e2e['latency_p50_seconds']:.3f}s` and p95
`{e2e['latency_p95_seconds']:.3f}s` after model load. Of {len(wrong)} synthetic errors,
`misc_uncat` safe fallback accounted for {fallback_errors.get('misc_uncat', 0)} rather than forcing
an unrelated category.

The successful Airflow run was
`{airflow_selection.get('orchestration', {}).get('airflow_dag_run_id') or 'not-recorded'}`. Airflow
generated one fixed Noto-font manifest and both Linux and Mac runtimes evaluated those exact image
hashes. An earlier Apple-font synthetic run scored lower, showing that fixture typography affects
OCR tests. None of these synthetic runs establishes real-photo accuracy.

![Candidate metrics](assets/candidate-top1.svg)

![Per-class F1](assets/hybrid-per-class-f1.svg)

![Confusion matrix](assets/hybrid-confusion.svg)

## Data and Provenance

| Source | Rows/images | Coverage | License/provenance | Use |
|---|---:|---:|---|---|
| Open Images V7 validation | {sum(openimages['counts'].values())} images | {openimages['covered_leaf_count']}/33 | Annotations CC BY 4.0; selected image records checked as CC BY 2.0 | Visual proxy and frozen head |
| `{source['dataset_id']}` revision `{source['revision']}` | {text_quality['source_counts']['hf_us_bank_transaction_categories_v2']:,} rows | Broad weak mapping | MIT; SHA-256 `{source['sha256']}` | Text weak supervision |
| CashLog template generator | {text_quality['source_counts']['cashlog_template_generator_v1']:,} rows | 33/33 | Project generated | OCR lexicon and routing coverage |
| Rendered Korean receipts | {e2e['samples']} images | 33/33 | Project generated | End-to-end contract test |

Open Images collection dropped `{openimages['ambiguous_cross_leaf_images_dropped']:,}` images whose
source labels mapped to more than one leaf. No download failed. The Openverse smoke source is not
used for training because anonymous API collection hit 401/429 limits and candidates remained
pending human review. PD12M discovery was also excluded after the dataset API returned 500/index
loading failures.

![Text class counts](assets/text-class-counts.svg)

## Input and Output Contract

Request: `POST /analyze-image`, multipart field `image`, or JSON containing `imageBase64`,
`mimeType`, and optional `filename`. Accepted signatures are JPEG, PNG, WebP, HEIC, and HEIF;
maximum size is 10 MiB. Declared MIME, extension, and binary signature must agree.

Response invariants:

- `recommended_category` is one of the exact 33 leaf IDs.
- `top_categories` contains up to three `{{category, confidence}}` entries.
- `need_user_check=true` means the client must not silently commit the first result.
- `error_code=LOW_CONFIDENCE` accompanies guarded decisions.
- `evidence` contains OCR lines, matched terms, component Top-3 values, image quality, fallback reason,
  and decision margin for debugging. This field should be removed or redacted at the public gateway
  if operational evidence is not intended for clients.

## Model Selection

The selected candidate is `{selection['selected_candidate']}` because the learned visual head beats
zero-shot on the same source-group holdout, all artifacts match pinned SHA-256 values, every leaf is
routable, and the full integration test passes the configured synthetic safety gates. It is deployed
only in guarded recommendation mode.

Production promotion is denied because `real_cashlog_holdout` is missing. Required frozen-holdout
gates are Top-1 >= 80%, Top-3 >= 95%, macro F1 >= 75%, minimum per-leaf recall >= 60%, ECE <= 8%,
false auto-confirm <= 2%, and p95 <= 3 seconds. A production promotion must be explicit even after
these gates pass.

## Failure and Recovery Log

| Failure | Root cause | Action |
|---|---|---|
| EfficientNet epoch 25 stopped | Airflow heartbeat timeout after best Top-1 88.60% on the old four-leaf task | Metrics retained; old run rejected for 33-leaf use |
| ConvNeXt exited 137 | Host memory pressure/OOM | Removed from current candidate; frozen embeddings + linear head used |
| Initial MLflow artifact upload failed | Server advertised read-only `/workspace` artifact URI | Enabled MLflow artifact proxy with `mlflow-artifacts:/` |
| Openverse full collection stopped | Anonymous 401/429 | Pending smoke rows excluded; Open Images official metadata used |
| PD12M discovery failed | HF dataset API 500/index loading | Excluded; failure documented |
| Generic receipt predicted `finance_fee` | OCR lost every semantic line and text model learned generic payment tokens | Added normalized display-name matching and `misc_uncat` safe fallback |
| Airflow vision-head attempt exited 137 | All 1,068 augmented PIL views were retained in memory before embedding | Replaced eager loading with bounded batch streaming and reduced Airflow batch to 4 |

## Monitoring and Operations

- Airflow: `http://127.0.0.1:8080`, DAG `cashlog33_training_pipeline`
- MLflow: `http://127.0.0.1:5500`, experiment `cashlog33-hybrid-v2`
- Jenkins: `http://127.0.0.1:8081`; credential ID `airflow-local-basic`
- API: `http://127.0.0.1:8010/health`
- Training progress: each text candidate writes `progress.json`; Airflow task logs provide stage state;
  MLflow stores parameters, metrics, status, and artifacts.

Airflow writes to `airflow_latest` candidate paths and never overwrites the pinned serving config.
Jenkins triggers and polls Airflow through its internal API and does not mount the Docker socket.

## Limitations and Robustness

- There is no frozen real CashLog photo holdout yet; no current percentage is a real app accuracy claim.
- Open Images labels identify objects, not financial intent. A phone photo cannot distinguish buying a
  phone from paying a phone bill without receipt/merchant context.
- Synthetic text and rendered receipts share vocabulary with the lexicon and therefore overestimate
  OCR/text performance.
- Current inference is CPU-oriented and loads roughly 1.4 GiB of SigLIP2 weights. Phone ONNX export
  is not the selected serving path; the Mac/home worker should run inference behind the private link.
- User correction events must include model version, Top-3, chosen leaf, image consent/retention state,
  and a stable de-identified sample ID before they can enter retraining.

## Next Evidence Required

1. Freeze at least 330 consented real photos, 10 or more for every leaf; 30 per leaf is preferred.
2. Keep the test split sealed and group repeated merchants/users to prevent leakage.
3. Run the same evaluator and store overall, per-leaf, calibration, latency, and auto-confirm metrics.
4. Review the lowest-recall leaves, add only licensed or consented training data, retrain an isolated
   candidate, and compare on the unchanged holdout.
5. Enable automatic confirmation only after the real-photo gates pass and the release is approved.

## 33-Leaf Coverage

{markdown_table(categories, text_quality)}
"""
    (output / "REPORT.md").write_text(report_md, encoding="utf-8")

    decision_rows = "".join(
        f"<tr><td>{html.escape(item['name'])}</td><td class={'pass' if item['passed'] else 'fail'}>{'PASS' if item['passed'] else 'FAIL'}</td><td>{html.escape(str(item['actual']))}</td><td>{html.escape(str(item['required']))}</td><td>{html.escape(item['scope'])}</td></tr>"
        for item in selection["gates"]
    )
    metric_rows = "".join(
        [
            f"<tr><td>Vision zero-shot</td><td>Open Images proxy, 23 leaves</td><td>{pct(vision['test_zero_shot_same_split']['top1_accuracy'])}</td><td>{pct(vision['test_zero_shot_same_split']['top3_accuracy'])}</td></tr>",
            f"<tr><td>Vision linear head</td><td>Same proxy split, 23 leaves</td><td>{pct(vision['test_linear_head']['top1_accuracy'])}</td><td>{pct(vision['test_linear_head']['top3_accuracy'])}</td></tr>",
            f"<tr><td>Serving hybrid</td><td>Mac runtime, fixed Noto fixtures, 33 leaves</td><td>{pct(e2e['top1_accuracy'])}</td><td>{pct(e2e['top3_accuracy'])}</td></tr>",
            f"<tr><td>Airflow candidate</td><td>Linux runtime, same fixtures, 33 leaves</td><td>{pct(airflow_e2e['top1_accuracy'])}</td><td>{pct(airflow_e2e['top3_accuracy'])}</td></tr>",
        ]
    )
    html_report = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CashLog 33 Model Report</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;color:#18202b;background:#fff;font:15px/1.55 system-ui,-apple-system,sans-serif}} header{{padding:44px 5vw 34px;background:#111827;color:#fff;border-bottom:5px solid #f97316}} h1{{margin:0 0 8px;font-size:34px;letter-spacing:0;overflow-wrap:anywhere}} header p{{max-width:920px;margin:0;color:#d1d5db;overflow-wrap:anywhere}} main{{max-width:1180px;margin:auto;padding:34px 24px 64px}} section{{padding:26px 0;border-bottom:1px solid #dfe4ea}} h2{{margin:0 0 14px;font-size:23px;letter-spacing:0}} .status{{display:inline-block;max-width:100%;padding:5px 9px;border-radius:4px;background:#fef3c7;color:#78350f;font-weight:700;overflow-wrap:anywhere}} .callout{{border-left:5px solid #dc2626;background:#fef2f2;padding:14px 16px;max-width:980px}} table{{width:100%;border-collapse:collapse;margin:14px 0}} th,td{{padding:9px 10px;border:1px solid #d7dde5;text-align:left;vertical-align:top}} th{{background:#f4f6f8}} .pass{{color:#047857;font-weight:700}} .fail{{color:#b91c1c;font-weight:700}} img{{display:block;width:100%;height:auto;border:1px solid #dfe4ea;margin:18px 0}} code{{background:#eef2f7;padding:2px 5px;border-radius:3px}} .metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:#d7dde5;border:1px solid #d7dde5}} .metric{{background:#fff;padding:16px}} .metric b{{display:block;font-size:24px}} @media(max-width:720px){{header{{padding:32px 20px 26px}} main{{padding:24px 20px 48px}} .metrics{{grid-template-columns:1fr 1fr}} table{{display:block;overflow:auto}} h1{{font-size:28px}}}}
</style></head><body>
<header><h1>CashLog 33-Leaf Hybrid Model</h1><p>Reproducible training, evaluation, serving, and promotion evidence. Generated {html.escape(generated_at)}.</p></header>
<main>
<section><span class="status">{html.escape(selection['decision'])}</span><h2>Decision</h2><p class="callout">The candidate is available for guarded Top-3 recommendation. It is not production-accuracy certified because a frozen real-photo holdout is missing. Automatic confirmation remains disabled.</p></section>
<section><h2>Measured Results</h2><div class="metrics"><div class="metric">Visual proxy Top-1<b>{pct(vision['test_linear_head']['top1_accuracy'])}</b>23 leaves</div><div class="metric">Airflow synthetic Top-1<b>{pct(airflow_e2e['top1_accuracy'])}</b>33 leaves</div><div class="metric">Airflow synthetic Top-3<b>{pct(airflow_e2e['top3_accuracy'])}</b>33 leaves</div><div class="metric">Linux CPU p95<b>{airflow_e2e['latency_p95_seconds']:.3f}s</b>after load</div></div><table><thead><tr><th>Candidate</th><th>Scope</th><th>Top-1</th><th>Top-3</th></tr></thead><tbody>{metric_rows}</tbody></table><img src="assets/candidate-top1.svg" alt="Candidate Top-1 chart"></section>
<section><h2>Promotion Gates</h2><table><thead><tr><th>Gate</th><th>Result</th><th>Actual</th><th>Required</th><th>Scope</th></tr></thead><tbody>{decision_rows}</tbody></table></section>
<section><h2>Per-Class Evidence</h2><img src="assets/hybrid-per-class-f1.svg" alt="Per-class F1"><img src="assets/hybrid-confusion.svg" alt="Confusion matrix"></section>
<section><h2>Data Coverage</h2><p>{text_quality['samples']:,} text rows cover all 33 leaves. Open Images contributes {sum(openimages['counts'].values())} license-traceable visual proxy images across 23 leaves. Synthetic evidence is explicitly separated from real-photo evidence.</p><img src="assets/text-class-counts.svg" alt="Text rows by category"></section>
<section><h2>Architecture and Operations</h2><p>SigLIP2 + learned visual head + Korean RapidOCR + TF-IDF SGD + deterministic lexicon, followed by weighted fusion and an unreadable-input fallback. Airflow isolates each candidate, MLflow records runs and artifacts, Jenkins triggers and monitors the DAG without a Docker socket.</p><p>The complete methodology, I/O contract, source checksums, failures, limitations, and all 33 labels are in <a href="REPORT.md">REPORT.md</a>. The exact run timeline and remaining operator actions are in <code>ml_docs/CASHLOG33_RUN_LOG.md</code>.</p></section>
</main></body></html>"""
    (output / "index.html").write_text(html_report, encoding="utf-8")
    summary = {
        "schema_version": 1,
        "generated_at": generated_at,
        "decision": selection["decision"],
        "production_eligible": selection["production_eligible"],
        "selected_candidate": selection["selected_candidate"],
        "metrics": {
            "vision_proxy": vision["test_linear_head"],
            "serving_synthetic_e2e": e2e,
            "airflow_synthetic_e2e": airflow_e2e,
        },
        "latest_airflow": airflow_selection.get("orchestration"),
        "scope_warning": "No frozen real CashLog photo holdout is available.",
    }
    (output / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
