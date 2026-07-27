# CashLog 33-Leaf Data Card

Version: 2026-07-27
Taxonomy: 33 leaves, `configs/cashlog/categories.json`

## Intended Use

This dataset collection supports a CashLog category recommendation model for product,
receipt, merchant, and transaction evidence. It is intended to propose Top 3 expense
categories. It is not intended to infer identity, creditworthiness, medical status, or
other sensitive attributes.

## Collected Sources

| Dataset | Acquisition | Rows/images | Coverage | License and status |
|---|---|---:|---:|---|
| US bank transaction categories v2 | Revision-pinned Hugging Face file API | 60,000 mapped rows | Weak mapping across available leaves | MIT; entirely synthetic source data |
| CashLog text templates | Deterministic local generator | 15,840 rows | All 33 leaves | Project-generated; integration/training aid |
| Open Images V7 validation | Google metadata files and image URLs | 411 images | 23 leaves | Annotations CC BY 4.0; selected images individually carry CC BY 2.0 metadata |
| Openverse smoke collection | Openverse API | 61 images | 31 leaves | 55 CC BY, 4 CC0, 2 PDM; weak query labels, train-only |
| Product Opener APIs | Open Food/Beauty/Pet Food Facts APIs | 972 rows downloaded across runs; 678 unique accepted | Grocery, beverage, beauty, pet | CC BY-SA 3.0 image metadata; product-type weak labels, train-only |
| UECFood256 | Local downloaded archive and project override map | 31,395 images | `meal_dining`, `meal_cafe` | Research dataset; bundled README states no redistribution license, so data/model redistribution is not assumed |
| CashLog actual | Consented import and manual labeling | 2 images | `meal_dining` | Private, human-approved, train-only |
| CashLog receipt fixtures | Deterministic local renderer | 99 images | 3 per each of 33 leaves | Project-generated; E2E test only |
| CashLog OCR/category synthetic v1 | Deterministic renderer from versioned text manifest | 112,200 full-page images | 3,400 per each of 33 leaves | Project-generated; synthetic training/validation only |
| CashLog OCR/category synthetic v2 | Four layouts, three fonts, OCR corruption and capture degradation | 508,200 full-page images | 15,400 per each of 33 leaves | Project-generated; 501,600 train, proxy holdouts only |
| CashLog OCR recognition v1 | Deterministic line crops from synthetic OCR boxes | 130,000 crops | Korean/English receipt lines | Project-generated; PaddleOCR recognition format |
| CORD v2 | Pinned Hugging Face Hub API download | 1,000 receipts | OCR word boxes and receipt layout | CC BY 4.0; OCR-only, never CashLog category truth |

The all-data text build contains 75,840 rows: 59,685 train, 8,067 validation, and
8,088 test. It keeps all 60,000 source rows that map to an expense leaf and all
15,840 generated rows. The source CSV has 68,000 rows; the remaining 8,000
income/transfer rows have no valid target in the expense-only 33-leaf taxonomy.
Counts and provenance are regenerated in
`data/processed/cashlog33/text/all_v1/quality_report.json`.

The original visual inventory before the OCR expansion is 31,869 images:
31,395 UECFood images,
411 Open Images images, 61 Openverse images, and 2 actual images. These are not
flattened into one misleading task. UECFood trains a prepared-food specialist;
the other 474 images train the SigLIP2 33-leaf visual head. The specialist only
redistributes probability already assigned to meal leaves.

The v2 expansion contains 501,600 balanced full-page training images and 6,600
held-out synthetic pages. Together with the retained v1 data, original visual
sources, and 678 unique Product Opener images, the local category-train-capable
image inventory is 639,747. The selected v2 text training manifest combines all
59,685 previous training rows with all 501,600 v2 rows for 561,285 training rows.
There are also 120,000 v1 recognition training crops. CORD contributes 800 real photographed
receipts to OCR training. Including validation/test files, the expansion contains
621,400 synthetic full-page images and 130,000 line crops. Synthetic images and CORD's
Indonesian domain labels do not enter the real-photo production holdout.

The Open Images collector dropped 4,058 images whose source labels mapped to multiple
CashLog leaves and recorded zero download failures. Ten leaves that cannot be
represented reliably as generic objects are intentionally excluded from the visual
proxy: housing rent, housing fee, housing utility, internet/TV, shows, three finance
leaves, unclassified, and other.

## API Acquisition Rules

- Pin source revision or metadata URL and store SHA-256, retrieval time, and license.
- Store attribution and original source URL for every third-party image.
- Never silently accept an ambiguous cross-leaf mapping.
- Treat source labels as proxy labels; they do not become real CashLog truth.
- Do not place downloaded datasets, personal images, model weights, or secrets in Git.
- A collector failure must leave a summary rather than changing an existing approved
  manifest in place.

Openverse full expansion encountered anonymous `401` and `429` responses. All 61
successfully downloaded and licensed smoke rows are retained as explicit weak
train-only labels; they never enter validation or test. PD12M dataset index access
returned server/index errors and supplied no local data to train.

The bounded Product Opener collector encountered transient HTTP `503` responses and
completed with retries and a resumable manifest rather than bypassing rate limits.
Across two runs it downloaded 972 rows. Merge validation removed 292 repeated product
IDs and two cross-leaf image conflicts, leaving 678 unique train-only weak labels.
The second grocery search failed with a recorded `503`; the 98 previously accepted
grocery images were preserved.
Future bulk expansion uses the official Open Food Facts product export and AWS Open
Data image/OCR bucket, as required by the provider for bulk acquisition.

## Split and Leakage Policy

- Text examples use deterministic `group_key` assignment so variations of one template
  or source transaction do not cross train, validation, and test.
- Visual examples use source-group splits. Auto-approved examples are train-only and
  cannot establish holdout accuracy.
- The real holdout uses a de-identified `group_id` for user/session/merchant grouping.
- Exact image SHA-256 duplicates and perceptual near-duplicates must not cross splits.
- Holdout labels require manual approval and must never be rewritten from predictions.

The accepted real-photo manifest contract is
`configs/cashlog/real_holdout.schema.json`. Validation is performed by
`scripts/validate_cashlog33_real_holdout.py` before evaluation.

## Real Data Required for Promotion

The current blocker is a consented, manually labeled set of at least 330 real CashLog
photos, with at least 10 independent examples for every leaf. A useful first training
pool is larger than this holdout; the 330-photo set is reserved for final evaluation
and must not be used for model or threshold tuning.

For each real sample, collect:

- Random `sample_id`, correct `leaf_id`, relative object path, and SHA-256.
- De-identified group ID for leakage prevention.
- `owner_approved` or licensed consent status.
- Manual review status and optional capture timestamp.
- No name, phone number, card number, address, or raw authentication token.

Before retention, redact receipt PII where possible. Store images in private object
storage, encrypt at rest, use short-lived backend URLs, and define a deletion/retention
period. User category corrections may be stored without the image unless the user
explicitly consents to image retention.

## Feedback Contract

The app records every explicit photo-category confirmation according to
`configs/cashlog/correction_event.schema.json`: model and taxonomy versions, proposed
Top 3, selected leaf, timestamp, confirmation source, review state, and separate
image-retention consent. Supabase RLS forces client writes to `pending`; only the
private review/export process can approve data. The model worker remains stateless
and does not write directly to Supabase. See `ml_docs/CASHLOG33_FEEDBACK_LOOP.md`.

## Known Limitations

- No frozen real-photo holdout currently exists, so production accuracy is unknown.
- Synthetic text accuracy is inflated by templates and broad label mappings.
- The general visual head has 31 train leaves after weak Openverse/Product Opener
  additions, but its frozen proxy holdout still covers only 23 leaves.
- UECFood contains prepared food but no trustworthy grocery or beverage expense
  labels. All images train the dining/cafe specialist.
- Visual support is sparse for `health_gym` and `gift_event` in the current proxy.
- OCR performance depends on blur, glare, crop, typography, and receipt language.
- The current RapidOCR baseline has synthetic CER `0.1422`; the new OCR training
  data is prepared, but a fine-tuned OCR model has not passed replacement gates yet.
- Current confidence calibration fails the production ECE threshold on synthetic E2E.
- Merchant and product distributions will drift after release.

These limitations are why all current responses require user confirmation.
