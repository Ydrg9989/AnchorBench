# P3 — Tool realism ablation

Addresses reviewer **REVIEWER-3** by varying two tool-realism axes:

- **Elicited**: a synthetic assistant turn announces the lookup before the tool call, framing the response as model-initiated.
- **Noisy**: the tool response wraps the anchor value in a realistic envelope with extra metadata fields (`confidence`, `freshness_days`, `sample_size_n`, `request_timestamp_ms`, etc.) to test salience-of-number.

## Mean UAI across the panel

| Relevance | Baseline | Elicited | Noisy |
|---|---:|---:|---:|
| Plausible | 0.25 | 0.23 | 0.25 |
| Irrelevant | 0.20 | 0.19 | 0.21 |

## Interpretation

- If `Elicited` ≈ `Baseline`, the published Tool number is robust to the provenance signal (and the controlled externally-injected setup is a fair test).
- If `Noisy` < `Baseline` plausibly, the published Tool number slightly overstates the in-practice anchoring effect because realistic tool responses dilute the anchor's salience.
- If `Noisy` ≈ `Baseline`, anchoring is driven by the numeric value regardless of surrounding metadata — the bias is real even in production-style tool envelopes.
