# CashLog 33 Model Selection

- Generated: `2026-07-17T11:29:23.361112+00:00`
- Candidate: `cashlog33-hybrid-v1.1-fast`
- Decision: **guarded_integration_candidate**
- Production eligible: **false**

## Gate Results

| Gate | Result | Actual | Required | Scope |
|---|---:|---:|---:|---|
| taxonomy_leaf_count | PASS | `33` | `33` | contract |
| artifact_sha256 | PASS | `True` | `True` | supply-chain |
| vision_head_beats_zero_shot_top1 | PASS | `0.7926829268292683` | `> 0.5975609756097561` | Open Images proxy |
| synthetic_33_leaf_coverage | PASS | `33` | `33` | synthetic I/O |
| synthetic_top1 | PASS | `0.98989898989899` | `>= 0.95` | synthetic I/O |
| synthetic_top3 | PASS | `0.98989898989899` | `>= 0.90` | synthetic I/O |
| synthetic_false_auto_confirm | PASS | `0.0` | `<= 0.02` | synthetic I/O |
| synthetic_latency_p95 | PASS | `0.27766522530000654` | `<= 3.0` | local mps synthetic I/O |
| real_cashlog_holdout | FAIL | `missing` | `frozen manual set: >=10 photos x 33 leaves` | real holdout |

## Decision Note

The hybrid is selected for guarded Top-3 recommendation serving. Auto-confirm stays disabled until the frozen real-photo holdout passes every production gate.

Proxy and synthetic metrics validate components and integration only. They do not replace a frozen, manually labeled CashLog photo holdout.
