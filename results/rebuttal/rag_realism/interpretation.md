# P2 — RAG realism ablation

Addresses reviewer **REVIEWER-2**: the RAG suite varies document rank (1 vs 5), adds plausible distractors, and optionally exposes synthetic relevance scores.

## Mean UAI across the panel

| Relevance | Base (rank 2) | Rank 1 | Rank 5 + 2 distract. | Rank 5 + distract. + scores |
|---|---:|---:|---:|---:|
| Plausible | 0.08 | 0.10 | 0.13 | 0.03 |
| Irrelevant | 0.02 | 0.01 | 0.03 | 0.02 |

## Interpretation

If UAI is higher at Rank 1 than Rank 5, models privilege top-ranked retrieved documents (a property of realistic RAG systems) and the published RAG number is an intermediate estimate. If UAI drops when distractors are added, the anchored document's salience is reduced by surrounding context — meaning real-world RAG pipelines with diverse retrieval would observe weaker anchoring than the controlled benchmark. The R5+D+Sc column tests whether explicit relevance scores let models down-weight the anchor when its retrieval score is mid-pack.
