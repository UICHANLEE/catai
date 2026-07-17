# CashLog 33-Leaf Data Card

Version: 2026-07-17
Taxonomy: 33 leaves, `configs/cashlog/categories.json`

## Intended Use

This dataset collection supports a CashLog category recommendation model for product,
receipt, merchant, and transaction evidence. It is intended to propose Top 3 expense
categories. It is not intended to infer identity, creditworthiness, medical status, or
other sensitive attributes.

## Collected Sources

| Dataset | Acquisition | Rows/images | Coverage | License and status |
|---|---|---:|---:|---|
| US bank transaction categories v2 | Revision-pinned Hugging Face file API | 18,669 used rows | Weak mapping across available leaves | MIT; entirely synthetic source data |
| CashLog text templates | Deterministic local generator | 15,840 rows | All 33 leaves | Project-generated; integration/training aid |
| Open Images V7 validation | Google metadata files and image URLs | 411 images | 23 leaves | Annotations CC BY 4.0; selected images individually carry CC BY 2.0 metadata |
| CashLog receipt fixtures | Deterministic local renderer | 99 images | 3 per each of 33 leaves | Project-generated; E2E test only |

The text build currently contains 34,509 rows: 26,794 train, 3,839 validation, and
3,876 test. Counts and provenance are regenerated in
`data/processed/cashlog33/text/v2/quality_report.json`.

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

Openverse was tested as an API source but anonymous collection encountered `401` and
`429` responses; its smoke output is not used for the selected model. PD12M dataset
index access returned server/index errors and was also excluded.

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

The app/backend records corrections according to
`configs/cashlog/correction_event.schema.json`: model version, proposed Top 3, selected
leaf, timestamp, source of correction, and separate image-retention consent. The
React Native app sends this to NestJS; NestJS validates and stores it under RLS. The
model worker is stateless and does not write directly to Supabase.

## Known Limitations

- No frozen real-photo holdout currently exists, so production accuracy is unknown.
- Synthetic text accuracy is inflated by templates and broad label mappings.
- Open Images covers only 23 leaves and contains objects, not Korean expense context.
- Visual support is sparse for `health_gym` and `gift_event` in the current proxy.
- OCR performance depends on blur, glare, crop, typography, and receipt language.
- Current confidence calibration fails the production ECE threshold on synthetic E2E.
- Merchant and product distributions will drift after release.

These limitations are why all current responses require user confirmation.
