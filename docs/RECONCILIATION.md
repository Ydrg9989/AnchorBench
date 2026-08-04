# Reconciliation ledger

Every known divergence between what the paper reports, what the committed
artifacts contain, and what the current code produces.

The rule for this file: divergences get **recorded**, not silently repaired.
The paper is accepted, so its numbers are frozen; the job is to make the gap
between paper and code explicit and explained. A row leaves this file only
when it is resolved, and the resolution is recorded with it.

Baseline for all comparisons: tag `colm2026-camera-ready`.

Status legend — **OPEN** needs a decision · **ACCEPTED** understood, no action
· **FIXED** resolved, with the fixing commit named.

---

## D1 — History regenerates at 120 items instead of 360

| | |
|---|---|
| **Status** | OPEN — fix belongs in Stage 2 |
| **Affects** | `datasets/anchorbench_history_core/`, and any attempt to reproduce the History suite |
| **Severity** | High for reproduction; **zero** for published numbers |

### What happens

`python -m anchorbench.data.generate --suite history --size core --seed 42` —
the documented recipe — produces **120 itemspecs and 600 core promptviews**.
The committed dataset, which is what the paper was run on, has **360 and
1,800**.

The 120 regenerated `item_id`s are a subset of the committed 360, but their
*content* differs too: `theta` differs in 108/120 rows, `evidence_structured`
in 110/120, and the gold answer `y_star` in 109/120. Only the first cell
survives, because the RNG stream diverges as soon as the item count per cell
changes.

### Root cause

`generate.py:63-67` applies one `n_per_cell` to every suite:

```python
_SIZE_PARAMS = {"core": {"n_per_cell": 10, ...}}
```

The offset-grid suites multiply that by the three anchor offsets
`{15, 25, 40}` (`itemspec_gen.py:47`):

```
external / icl / icl_dist / rag / tool:  6 domains x 2 difficulties x 3 offsets x 10 = 360
history (no offset dimension):           6 domains x 2 difficulties x           10 = 120
```

`generate_history_itemspecs` (`itemspec_gen.py:415`) is documented as
"domain × difficulty grid, no offset dimension" and fixes the anchor at
theta ± 25. So History needs `n_per_cell = 30` to reach the same 360 items,
and the `core` preset does not give it that.

### Verified

`--n_per_cell 30` reproduces the committed dataset exactly:

```bash
python -m anchorbench.data.generate --suite history --size core --seed 42 \
    --n_per_cell 30 --out_dir /tmp/history_check
```

| file | matches committed |
|---|---|
| `promptviews.jsonl` | yes, byte-for-byte |
| `promptviews_core.jsonl` | yes, byte-for-byte |
| `promptviews_ablation.jsonl` | yes, byte-for-byte |
| `itemspecs.jsonl` | yes, apart from `generator_version` (see D2) |

So the committed History data **is** reproducible; the size preset is what is
wrong, not the generator and not the data.

### Disposition

No paper number is affected — the paper used the 360-item dataset, which is
committed and verified reproducible. But anyone following the documented
command silently gets a third of the benchmark, so this is a real defect.

Stage 2 fix: make the `core` preset yield 360 items for every suite, e.g. by
scaling `n_per_cell` for suites without an offset dimension rather than
hard-coding 30 at the call site. Verification is exact — regenerate all six
suites and compare against the committed hashes.

**Do not** regenerate `datasets/anchorbench_history_core/` as part of that
fix. The committed data is the paper's ground truth.

---

## D2 — `generator_version` differs on every regenerated itemspec

| | |
|---|---|
| **Status** | ACCEPTED — working as designed |
| **Affects** | `itemspecs.jsonl` in every dataset |

`generate.py:70` stamps each itemspec with `git rev-parse --short HEAD`.
Regenerating on a different commit therefore changes one field in all 360
rows, and with it the file hash. The committed datasets carry `a337b4d`;
regenerating at `99dac13` writes `99dac13`.

This is provenance, not data. It is also why `itemspecs.jsonl` hashes cannot
be used as a reproducibility check on their own — compare
`promptviews*.jsonl`, or compare itemspecs field-by-field ignoring
`generator_version`.

Note that `render_version` (`itemspec_gen.py:231,481`) is a *different* field,
hard-coded to `"2.1.0"`, and is genuinely data-affecting: changing it changes
every dataset hash. It must not be "synced" to the package version.

---

## D3 — Stage 1.1 result: five of six suites reproduce exactly

| | |
|---|---|
| **Status** | ACCEPTED — recorded as the reproducibility baseline |

Regenerated at `--size core --seed 42` and compared against the committed
datasets:

| suite | `promptviews*.jsonl` | `itemspecs.jsonl` |
|---|---|---|
| external | identical | identical apart from D2 |
| icl | identical | identical apart from D2 |
| icl_dist | identical | identical apart from D2 |
| rag | identical | identical apart from D2 |
| tool | identical | identical apart from D2 |
| history | identical **with `--n_per_cell 30`** (D1) | identical apart from D2 |

The prompts the models actually saw are reproducible from current code for all
six suites. This is the strongest reproducibility result in the audit.

---

## D4 — The CI at `findings.tex:107` was not produced by a bootstrap

| | |
|---|---|
| **Status** | **FIXED** in code; paper wording is an author decision (see below) |
| **Affects** | `tab:stats_inference` bottom row; the interval quoted in main-text prose |
| **Severity** | Downgraded to Low after measurement — the published number is sound |

### Measured outcome (Stage 1.6)

| | range | 95% CI |
|---|---|---|
| **Published** (`findings.tex:107`, `appendix.tex:1077`) | 0.40 | **[0.20, 0.62]** |
| Broken sampler, i.e. what the code emitted before the fix | 0.40 | [0.23, 0.57] |
| **Corrected bootstrap**, seed 42, B=2000 | 0.40 | **[0.20, 0.60]** |

The point estimate is unaffected: it is computed from the data, not the
bootstrap, and reproduces as 0.3962 → **0.40** exactly as published.

**The published interval is consistent with a correct bootstrap; the code was
what had drifted.** Across 200 seeds at B=2000 the corrected upper bound
rounds to 0.60 in 47% of runs and to 0.62 in 7%, and never falls below 0.59 —
whereas the broken sampler's 0.57 lies outside that entire range. The
published lower bound of 0.20 matches the corrected value exactly.

This also revises the earlier suspicion (recorded when the audit was written)
that fixing the sampler would *invalidate* the published interval. It does the
opposite: it moves the code back into agreement with the paper, to within
seed noise.

Direction of the error is worth noting: the subset sampler **understated**
uncertainty. Retaining ~6.5 of 10 distinct models and averaging them
unweighted never produces the heavily-reweighted draws a bootstrap depends
on, so the interval came out too narrow. The fix widens it, which is the
conservative direction.

### Fixed

`tables_appendix.py` now resamples with replacement and indexes per-cell
values so repeats count repeatedly. Regenerating `tab_stats_inference.tex`
changes exactly one line and nothing else in the table.

`tests/test_stats_inference_bootstrap.py` pins it by recomputing a correct
bootstrap from scratch and comparing, rather than asserting a literal, plus a
second test asserting the two samplers still disagree so the first cannot
quietly lose its teeth. Verified non-vacuous against the reintroduced bug.

### Open: what the paper should say

Published **[0.20, 0.62]** vs regenerated **[0.20, 0.60]**. Both are correct
bootstrap outcomes; they differ only by which seed was drawn. Options:

1. **Update the paper to [0.20, 0.60]** — the released code then reproduces
   the published number exactly, which is the point of this ledger. Two
   characters in `findings.tex:107` and one row in `appendix.tex:1077`;
   negligible pagination impact.
2. **Leave [0.20, 0.62]** — defensible, but a reader running the released
   code gets 0.60 and has no way to know why.

Recommendation: option 1.

`tables_appendix.py:263-270`:

```python
idx = rs.randint(0, len(ow_models), size=len(ow_models))
sample_models = [ow_models[i] for i in idx]
vals = [... for r in unified if ... r["model"] in sample_models ...]
```

`in sample_models` is a membership test, so a model drawn three times
contributes once. This is not resampling with replacement — it is a random
subset that retains each model with probability `1 - (1 - 1/10)^10 ≈ 0.651`.
The correct pattern appears 30 lines below in the same function
(`tables_appendix.py:295-296`, the Pearson CI), which indexes the arrays
directly.

Separately, the published interval **[0.20, 0.62]** is not reproducible from
any currently reachable combination of code and data: both
`unified_all_suites.json` files are unchanged since commit `9bf3fe2`, and
`build_stats_inference` is byte-identical to its pre-2.0 ancestor. Today's
code yields **[0.23, 0.57]** — still from the broken sampler. The published
interval came from a data snapshot that no longer exists.

Next step is measurement, not repair: correct the sampler, recompute, and
report the interval and whether Finding 1's qualitative claim survives,
before deciding whether to re-typeset.

---

## D5 — Two appendix tables cannot be regenerated

| | |
|---|---|
| **Status** | OPEN — reruns scheduled (Stage 3) |
| **Affects** | `tab:tool_plaintext`, `tab:history_matched` |

`tables_appendix.py` defaults `--tool_plaintext` and `--history_matched` to
`None`, in which case it emits `---` for two of four columns and a single row
respectively. The input directories used for the paper were never preserved,
and no directory under `results/` contains the `control_twostage` condition
that `tab:history_matched` needs.

Decision taken: rerun both and publish as a versioned addendum beside the
frozen PDF numbers, never substituting silently. vLLM inference is not
bitwise reproducible at temperature 0, so the reruns will not match the
published values exactly — the same effect already documented for the CoT
cell in `results/rebuttal/cot_replication/README.md`.

---

## D6 — Appendix tables drift from current generator output

| | |
|---|---|
| **Status** | OPEN — awaiting per-table disposition |

Roughly eight appendix tables no longer match what their generators emit.
Reported magnitudes range from below reporting precision (`p 0.99` vs `1.00`
in three per-suite tables; `tab:anchored_mae` by 0.01-0.06) to material
(`tab:stats_inference`, see D4). `verify.py`'s `PAPER_AMAE` tolerance is 0.20,
twenty times the observed drift, so verify passes while the table disagrees.

These figures come from the audit sweep and have **not yet been
re-measured** here. Stage 1.4 transcribes the published values into
`verify.py` so each one becomes a mechanically checked row rather than a
prose claim; this entry gets split per table at that point.

Structural note: every drifting table is *pasted inline* in `appendix.tex`,
and none of the 13 `\input{}`-ed tables drifted. Converting the pasted ones to
`\input{}` would retire this failure mode permanently.
