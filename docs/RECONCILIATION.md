# Reconciliation ledger

Every known divergence between what the paper reports, what the committed
artifacts contain, and what the current code produces.

The rule for this file: divergences get **recorded**, not silently repaired.
The paper is accepted, so its numbers are frozen; the job is to make the gap
between paper and code explicit and explained. A row leaves this file only
when it is resolved, and the resolution is recorded with it.

Baseline for all comparisons: tag `colm2026-camera-ready`.

Status legend — **OPEN** needs a decision · **ACCEPTED** understood, no action
needed · **RESOLVED** fixed, with the resolution recorded.

## Index

| | divergence | status | affects a published number? |
|---|---|---|---|
| [D1](#d1) | History regenerated at 120 items instead of 360 | RESOLVED | no |
| [D2](#d2) | `generator_version` differs on every regenerated itemspec | ACCEPTED | no |
| [D3](#d3) | Five of six suites reproduce exactly; History needs `--n_per_cell 30` | ACCEPTED | no |
| [D4](#d4) | The range CI was not produced by a bootstrap | RESOLVED | yes — corrected |
| [D5](#d5) | Two appendix tables cannot be regenerated | OPEN → Stage 3 | yes — reruns planned |
| [D6](#d6) | Inline appendix tables vs current output (measured) | RESOLVED | yes — 4 tables now generated |
| [D7](#d7) | `verify_anchored_mae` verified nothing at all | RESOLVED | no — check was dead |

`verify.py::KNOWN_DIVERGENCES` is **empty**, and all three verify modes
report zero mismatches. That is the goal state, not an unused mechanism: the
table still fails the run if an entry stops firing, so a divergence cannot be
recorded and then quietly swallowed by a tolerance.

How to re-measure:

```bash
python -m pytest tests/test_dataset_regeneration.py   # D1, D2, D3
python -m pytest tests/test_golden_artifacts.py       # generator self-consistency
python scripts/measure_paper_drift.py --verbose       # D6
python -m anchorbench.paper.verify --strict           # paper claims
```

---

<a id="d1"></a>

## D1 — History regenerates at 120 items instead of 360

| | |
|---|---|
| **Status** | **RESOLVED** — size preset fixed in Stage 2.2 |
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

### Resolved

`generate.py` now scales `n_per_cell` for suites without an offset dimension
(`_SUITES_WITHOUT_OFFSET_GRID`), so every suite reaches 360 items from
`--size core` alone. `--suite history --size core --seed 42` yields 360
itemspecs and 1,800 core promptviews, matching the published suite.

The committed datasets were **not** regenerated — they are the paper's ground
truth. `tests/test_dataset_regeneration.py` previously carried an explicit
`n_per_cell=30` for History to work around this; it no longer does, and all
six suites still reproduce byte-for-byte. That is what proves the fix changed
the recipe without disturbing the data.

---

<a id="d2"></a>

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

<a id="d3"></a>

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

<a id="d4"></a>

## D4 — The CI at `findings.tex:107` was not produced by a bootstrap

| | |
|---|---|
| **Status** | **RESOLVED** — code fixed, paper updated to match |
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

### Resolved in the paper

The authors chose to move the paper to the value the released code produces,
so a reader running `anchorbench tables` reproduces the published interval
exactly. Changed in two places:

- `sections/findings.tex:107` — the prose in Finding 1
- `sections/appendix.tex:1078` — the `tab:stats_inference` row

Rebuilt and verified: 34 pages, 0 errors, 0 overfull boxes, main text still
ending at the bottom of page 10 with the Ethics statement opening page 11.
The pre-change PDF remains reachable at tag `colm2026-camera-ready`.

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

<a id="d5"></a>

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

<a id="d6"></a>

## D6 — Appendix tables vs current generator output (measured)

| | |
|---|---|
| **Status** | **RESOLVED** — the four drifting tables are now generated, not pasted |
| **Measured with** | `python scripts/measure_paper_drift.py [--verbose]` |

The 13 `\input{}`-ed tables cannot drift by construction. Of the 14 tables
pasted inline in `appendix.tex`, **8 match current output exactly**, 4 carry
small numeric drift, and 2 are structurally incomplete (D5).

This supersedes the audit sweep's estimate of "roughly eight drifting
tables". Measurement moved three tables — `tab:app-external`,
`tab:app-rag`, `tab:app-tool` — from "drifted" to identical; the reported
`p 0.99` vs `1.00` difference is not present in the data.

| paper table | status | cells differing | max abs delta |
|---|---|---|---|
| `tab:app-external` | identical | 0/29 | — |
| `tab:app-history` | identical | 0/29 | — |
| `tab:app-icl` | identical | 0/29 | — |
| `tab:app-rag` | identical | 0/29 | — |
| `tab:app-tool` | identical | 0/28 | — |
| `tab:uai_summary` | identical | 0/35 | — |
| `tab:icl_dist` | identical | 0/40 | — |
| `tab:model-details` | no numeric cells | — | — |
| `tab:stats_inference` | **drift** | 5/29 | **0.19** |
| `tab:uai_distribution` | drift | 10/27 | 0.20 |
| `tab:anchored_mae` | drift | 8/10 | 0.06 |
| `tab:difficulty` | drift | 10/17 | 0.50 |
| `tab:history_matched` | structural | 10 rows vs 1 | see D5 |
| `tab:tool_plaintext` | structural | 4 cols vs 2 | see D5 |

### D6a — `tab:stats_inference` (was 5/29 cells, now 4/29)

The range-CI row is no longer among them; that was D4 and is resolved.

**ICL $p_{\mathrm{BH}}$ 0.62 → 0.81 is now also resolved**: it was the only
cell here with a delta above rounding noise, and it was quoted in main-text
prose at `sections/findings.tex:123`. Updated at both sites to the value the
released code produces. Both values are far from significance and the
sentence already read "n.s.", so no claim changed.

Four cells remain, every one of them a delta of exactly 0.01:

| cell | paper | current | also quoted in prose? |
|---|---|---|---|
| External CI lower | 0.12 | 0.13 | no |
| History CI lower | 0.11 | 0.12 | no |
| History $p_{\mathrm{BH}}$ | 0.01 | 0.02 | yes — `findings.tex:122` reads "$\approx 0.01$" |
| Pearson CI upper | $-$0.01 | $-$0.00 | yes — `findings.tex:193` and the Fig. 4 caption |

**Disposition:** deliberately left in place for now. Stage 1.4 transcribes
this table into `verify.py`, so these four become mechanically checked rows
rather than prose claims — they are the first real exercise of that
machinery. Align them when the `\input{}` conversion regenerates the table as
a block (Stage 2), or sooner if preferred; nothing here is at risk of
changing a claim.

Ruled out as the cause: the positional pairing in
`build_stats_inference` (`pls_arr[:n] - irr_arr[:n]`, which pairs by
position rather than by model). Checked directly — every suite has **zero**
cells where one of `uai_plaus` / `uai_irr` is present and the other is not,
so the truncation is a no-op and the arrays are aligned. It is fragile and
worth hardening in Stage 2, but it is not producing these numbers.

Remaining explanation is the same as D4: the paper's values came from a data
snapshot that no longer exists. Recomputing from the committed
`unified_all_suites.json` gives ICL raw $p = 0.8077$, which BH leaves
unchanged as the largest p-value — self-consistent with the current table.

**Disposition:** recommend updating `findings.tex:123` and the
`tab:stats_inference` row to the values the released code produces, for the
same reason D4 was updated. Author decision.

### D6b — `tab:uai_distribution` (10/27), `tab:anchored_mae` (8/10), `tab:difficulty` (10/17)

Max deltas 0.20, 0.06 and 0.50 respectively. All are at or below the
reporting precision of the surrounding prose, and none is quoted in the main
text or carries a claim that flips. `tab:difficulty`'s 0.5 is on an
accuracy percentage (66.0 → 65.5), i.e. half a point on a 0-100 scale.

**Disposition:** accept and document, or regenerate as a block if the
`\input{}` conversion below is adopted. No claim is at risk either way.

### Structural note

Every drifting table is *pasted inline* in `appendix.tex`; not one of the 13
`\input{}`-ed tables drifted. Converting the pasted ones to `\input{}`, with
a build step copying `outputs/tables/*.tex` into `COLM_camera_ready/tables/`,
would retire this whole failure mode. That is the single highest-leverage
change in the audit and belongs in Stage 2.

---

<a id="d7"></a>

## D7 — `verify_anchored_mae` verified nothing at all

| | |
|---|---|
| **Status** | **RESOLVED** — check repaired |
| **Affects** | `tab:anchored_mae` (Table 12), and the audit's reading of `PAPER_AMAE` |

The audit recorded that `PAPER_AMAE`'s tolerance of 0.20 was "twenty times
the actual drift, so verify passes while the table is wrong". Measuring it
turned up something worse: the check never ran.

`verify_anchored_mae` read `mae_irr` and `mae_plaus` from the unified
summaries. Those keys have never existed there — the only MAE field is
`mae_control`. Every suite therefore hit the `if not d_irr_list: continue`
branch, printed `skipped (no mae_irr/mae_plaus in data)`, and contributed
nothing. `PAPER_AMAE` was dead at *any* tolerance.

The deltas live in the per-record generations, which is where the table
itself gets them, via `tables_appendix.compute_delta_mae_by_suite`. The check
now calls that same function, so it verifies the numbers the table prints.

With the check live, all five suites report values matching
`scripts/measure_paper_drift.py` exactly, and the eight genuine divergences
(0.007–0.058) are recorded in `KNOWN_DIVERGENCES`.

`tests/test_paper_verify.py::test_anchored_mae_is_actually_computed` asserts
the check produces values and never re-enters the skipped branch.

### On `--strict`

`--strict` used to halve tolerances to 0.002 on these tables. That is below
the printed precision: values shown to 2 d.p. carry up to 0.005 of legitimate
rounding, so 0.002 flagged twelve rounding artefacts as disagreements. Both
tables now use `ROUNDING_2DP = 0.005` in either mode, which is the smallest
tolerance that is meaningful for a number printed to two decimals. Anything
above it is a real divergence and belongs in `KNOWN_DIVERGENCES`, not in a
wider tolerance.


---

## D6 resolution — the pasted tables are now generated

The four drifting tables (`tab:stats_inference`, `tab:uai_distribution`,
`tab:anchored_mae`, `tab:difficulty`) no longer carry pasted numbers.
`appendix.tex` `\input{}`s a generated body for each, produced by
`scripts/sync_paper_tables.py` from `outputs/tables/`.

**Only the `tabular` is generated.** The `\begin{table}` wrapper, float type,
`\caption` and `\label` stay in `appendix.tex`, because the captions carry
interpretation the generator has no business owning — "Hard items show
stronger discrimination on External and RAG", the $n{=}14$/$n{=}13$ sample
sizes, "Overshoot concentrates in History (23.8\% for plausible)". A naive
whole-float `\input` would have deleted all of it. Generator owns the
numbers; author owns the prose.

Caption and prose values that restated a drifted cell were updated with it:

| site | was | now |
|---|---|---|
| `tab:anchored_mae` caption, Tool | $-$0.65 | $-$0.67 |
| `tab:uai_distribution` caption, overshoot | 5–7\% | 4–7\% |
| `findings.tex:122`, History $p_{\mathrm{BH}}$ | $\approx$0.01 | $\approx$0.02 |
| `findings.tex:207` and `appendix.tex:1101`, Pearson CI | [$-$0.43, $-$0.01] | [$-$0.43, $-$0.00] |

Two caption claims were **checked and left alone** because they still hold:
the difficulty caption's "hard items show stronger discrimination on External
and RAG" (External 0.11→0.23, RAG 0.05→0.16, and History easy 0.33 > hard
0.23 as stated), and the 23.8\% History overshoot figure, recomputed from raw
records as exactly 23.8\%.

`tab:history_matched` and `tab:tool_plaintext` are deliberately **not**
converted: their input runs were never preserved, so the generator emits a
single row and `---` placeholders. Syncing them would replace published
numbers with blanks. They stay pasted until the D5 re-runs.

Guarded by `tests/test_golden_artifacts.py::test_paper_table_bodies_are_in_sync`,
which runs `sync_paper_tables.py --check` and fails if a committed body has
gone stale against the generator.
