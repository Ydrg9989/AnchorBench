# Pre-Run Validation Checklist

**Purpose:** Run this checklist before launching full model experiments. Every item must pass. Failures are blocking unless marked otherwise.

---

## 1. Dataset Schema Completeness

### 1.1 ItemSpec Schema [MUST]

For each pilot dataset (`datasets/anchorbench_*_pilot/itemspecs.jsonl`):

- [ ] Every record has all required fields: `item_id`, `suite`, `domain`, `template_family`, `theta`, `y_star`, `y_star_evidence`, `difficulty`, `anchors`, `evidence_structured`, `answer_space`, `render_version`
- [ ] `theta` is in [30, 70] for every item
- [ ] `y_star_evidence` is in [0, 100] for every item
- [ ] `y_star_evidence == round(mean(visible_evidence_values))` for every item
- [ ] `anchors.low` is in [0, 100] and `anchors.high` is in [0, 100]
- [ ] `anchors.low < theta < anchors.high` (for offset-stratified suites)
- [ ] `difficulty` is one of `"easy"`, `"hard"`
- [ ] `domain` is one of the 6 frozen domain IDs
- [ ] `suite` matches the expected suite for the dataset

```bash
PYTHONPATH=src python -c "
from anchorbench_v1.validators import validate_all
from anchorbench_v1.schema import ItemSpec, PromptView, read_jsonl
import sys

suites = {
    'external': 'datasets/anchorbench_external_pilot',
    'history': 'datasets/anchorbench_history_pilot',
    'icl': 'datasets/anchorbench_icl_pilot',
    'rag': 'datasets/anchorbench_rag_pilot',
    'tool': 'datasets/anchorbench_tool_pilot',
}
for suite, path in suites.items():
    specs = [ItemSpec(**r) for r in read_jsonl(f'{path}/itemspecs.jsonl')]
    views = [PromptView(**r) for r in read_jsonl(f'{path}/promptviews.jsonl')]
    errors = validate_all(specs, views, suite=suite)
    if errors:
        print(f'FAIL {suite}: {len(errors)} errors')
        for e in errors[:5]:
            print(f'  {e}')
        sys.exit(1)
    else:
        print(f'PASS {suite}: {len(specs)} items, {len(views)} views')
print('All validation passed.')
"
```

### 1.2 PromptView Schema [MUST]

For each pilot dataset (`datasets/anchorbench_*_pilot/promptviews.jsonl`):

- [ ] Every record has: `item_id`, `condition`, `prompt_text`, `prompt_hash`, `anchor_value`, `anchor_relevance`
- [ ] `condition` is one of: `control`, `irrelevant_low`, `irrelevant_high`, `plausible_low`, `plausible_high`
- [ ] `anchor_relevance` matches condition: control → `"none"`, irrelevant_* → `"irrelevant"`, plausible_* → `"plausible"`
- [ ] `anchor_value` is `null` for control; integer in [0, 100] for anchored conditions
- [ ] `prompt_text` contains the answer format instruction: "Return only a single integer 0–100 on the last line."

---

## 2. Paired-Condition Completeness [MUST]

For each dataset:

- [ ] Every `item_id` has exactly 5 prompt views (one per condition)
- [ ] No item is missing any condition
- [ ] No duplicate (item_id, condition) pairs exist

```bash
PYTHONPATH=src python -c "
from anchorbench_v1.schema import PromptView, read_jsonl
from collections import Counter

EXPECTED = {'control', 'irrelevant_low', 'irrelevant_high', 'plausible_low', 'plausible_high'}
datasets = [
    'datasets/anchorbench_external_pilot',
    'datasets/anchorbench_history_pilot',
    'datasets/anchorbench_icl_pilot',
    'datasets/anchorbench_rag_pilot',
    'datasets/anchorbench_tool_pilot',
]
for ds in datasets:
    views = read_jsonl(f'{ds}/promptviews.jsonl')
    by_item = {}
    for v in views:
        by_item.setdefault(v['item_id'], set()).add(v['condition'])
    issues = []
    for iid, conds in by_item.items():
        if conds != EXPECTED:
            issues.append(f'{iid}: got {conds}')
    pairs = Counter((v['item_id'], v['condition']) for v in views)
    dupes = {k: c for k, c in pairs.items() if c > 1}
    if issues or dupes:
        print(f'FAIL {ds}: {len(issues)} incomplete, {len(dupes)} duplicates')
    else:
        print(f'PASS {ds}: {len(by_item)} items, all 5 conditions present')
"
```

---

## 3. Count Balance [MUST]

### 3.1 Domain Balance

- [ ] Each suite has equal item count per domain (within each suite)
- [ ] All 6 domains are represented in every suite

### 3.2 Difficulty Balance

- [ ] Each suite has equal item count per difficulty level (easy vs hard)

### 3.3 Offset Balance (External, ICL, RAG, Tool only)

- [ ] Equal item count per offset value (15, 25, 40)

```bash
PYTHONPATH=src python -c "
from anchorbench_v1.schema import read_jsonl
from collections import Counter

FROZEN_DOMAINS = {'pricing_wtp', 'operations_time', 'transportation_logistics',
                  'resource_consumption', 'market_demographics', 'legal_policy'}

datasets = {
    'External': 'datasets/anchorbench_external_pilot',
    'History': 'datasets/anchorbench_history_pilot',
    'ICL': 'datasets/anchorbench_icl_pilot',
    'RAG': 'datasets/anchorbench_rag_pilot',
    'Tool': 'datasets/anchorbench_tool_pilot',
}
for name, path in datasets.items():
    specs = read_jsonl(f'{path}/itemspecs.jsonl')
    domains = Counter(s['domain'] for s in specs)
    diffs = Counter(s['difficulty'] for s in specs)
    dom_set = set(domains.keys())
    missing = FROZEN_DOMAINS - dom_set
    dom_balanced = len(set(domains.values())) == 1
    diff_balanced = len(set(diffs.values())) == 1
    status = 'PASS' if (not missing and dom_balanced and diff_balanced) else 'FAIL'
    print(f'{status} {name}: domains={dict(domains)}, difficulties={dict(diffs)}')
    if missing:
        print(f'  MISSING DOMAINS: {missing}')
"
```

---

## 4. Gold-Answer Validity [MUST]

- [ ] For every item, `y_star_evidence == round(mean(visible_evidence_values))`
- [ ] `y_star_evidence` is in [0, 100]
- [ ] Evidence values are in [0, 100]

```bash
PYTHONPATH=src python -c "
from anchorbench_v1.schema import read_jsonl
import statistics

datasets = [
    'datasets/anchorbench_external_pilot',
    'datasets/anchorbench_history_pilot',
    'datasets/anchorbench_icl_pilot',
    'datasets/anchorbench_rag_pilot',
    'datasets/anchorbench_tool_pilot',
]
for ds in datasets:
    specs = read_jsonl(f'{ds}/itemspecs.jsonl')
    errors = 0
    for s in specs:
        ev = s.get('evidence_structured', [])
        visible = [e['value'] for e in ev if e.get('value') is not None and e.get('available', True)]
        if not visible:
            continue
        expected = round(statistics.mean(visible))
        actual = s.get('y_star_evidence', s.get('y_star'))
        if actual != expected:
            errors += 1
            if errors <= 3:
                print(f'  {s[\"item_id\"]}: expected {expected}, got {actual}')
    status = 'PASS' if errors == 0 else 'FAIL'
    print(f'{status} {ds}: {errors} gold-answer mismatches out of {len(specs)} items')
"
```

---

## 5. Parser Sanity [MUST]

Verify the parser produces expected results on known inputs:

```bash
PYTHONPATH=src python -c "
from mitigation_eval.runner import parse_answer_int

tests = [
    ('42', '', 42),
    ('The answer is 73.', '', 73),
    ('Based on analysis, I estimate 55\n55', '', 55),
    ('Rating: 88\n', '', 88),
    ('I think it is about 101', '', None),  # out of range
    ('-5', '', None),  # negative
    ('No numeric answer here.', '', None),  # no number
    ('The value is 30. After reconsidering, the value is 65.', '', 65),  # last valid
]
all_ok = True
for text, prompt, expected in tests:
    result = parse_answer_int(text, prompt)
    status = 'PASS' if result == expected else 'FAIL'
    if result != expected:
        all_ok = False
    print(f'{status}: parse_answer_int({text!r}) = {result} (expected {expected})')
if all_ok:
    print('All parser tests passed.')
else:
    print('SOME PARSER TESTS FAILED.')
"
```

---

## 6. Deterministic Prompt Generation [MUST]

Regenerate datasets with the same seed and verify prompts are identical:

```bash
# Regenerate ICL pilot to a temp dir and compare
PYTHONPATH=src python -m anchorbench_v1.generate \
  --suites icl --size pilot --seed 42 \
  --out_dir /tmp/anchorbench_icl_check

diff <(sort datasets/anchorbench_icl_pilot/itemspecs.jsonl) \
     <(sort /tmp/anchorbench_icl_check/itemspecs.jsonl)

diff <(sort datasets/anchorbench_icl_pilot/promptviews.jsonl) \
     <(sort /tmp/anchorbench_icl_check/promptviews.jsonl)

# Should produce no diff. If it does, generation is non-deterministic.
rm -rf /tmp/anchorbench_icl_check
```

Repeat for all five suites.

---

## 7. Condition Naming Consistency [MUST]

- [ ] All prompt views use: `control`, `irrelevant_low`, `irrelevant_high`, `plausible_low`, `plausible_high`
- [ ] All eval scripts use the same condition set

```bash
PYTHONPATH=src python -c "
from anchorbench_v1.schema import read_jsonl

CONDITIONS = {'control', 'irrelevant_low', 'irrelevant_high', 'plausible_low', 'plausible_high'}

datasets = [
    'datasets/anchorbench_external_pilot',
    'datasets/anchorbench_history_pilot',
    'datasets/anchorbench_icl_pilot',
    'datasets/anchorbench_rag_pilot',
    'datasets/anchorbench_tool_pilot',
]
for ds in datasets:
    views = read_jsonl(f'{ds}/promptviews.jsonl')
    conds = set(v['condition'] for v in views)
    if conds == CONDITIONS:
        print(f'PASS {ds}: conditions = {sorted(conds)}')
    else:
        unexpected = conds - CONDITIONS
        missing = CONDITIONS - conds
        print(f'FAIL {ds}: unexpected={unexpected}, missing={missing}')
"
```

---

## 8. Artifact Anonymization Risks [MUST before submission]

Run before creating the submission archive:

```bash
# Check for author names / emails (replace with actual names to check)
rg -i "your_name_here\|your_email\|your_university" --type-add 'all:*' -t all .

# Check for internal paths
rg "/home/\|/Users/\|@.*\.edu\|@.*\.com" --type py --type md --type sh .

# Check for API keys
rg "sk-\|OPENROUTER_API_KEY\|OPENAI_API_KEY\|HF_TOKEN" .

# Check for .env files
find . -name ".env*" -not -path "./.gitignore"

# Check for git author info
git log --format='%an <%ae>' | sort -u

# Check for W&B references
rg "wandb\|weights.*biases\|w&b" --type py .
```

- [ ] No identifying author information found
- [ ] No internal server paths found
- [ ] No API keys or tokens found
- [ ] No .env files included
- [ ] Git history is clean or will be squashed for artifact
- [ ] W&B references are anonymized or removed

---

## 9. Manifest Consistency [MUST]

For each dataset, verify `manifest.json` matches actual file contents:

```bash
PYTHONPATH=src python -c "
from anchorbench_v1.schema import read_jsonl
import json

datasets = {
    'External': 'datasets/anchorbench_external_pilot',
    'History': 'datasets/anchorbench_history_pilot',
    'ICL': 'datasets/anchorbench_icl_pilot',
    'RAG': 'datasets/anchorbench_rag_pilot',
    'Tool': 'datasets/anchorbench_tool_pilot',
}
for name, path in datasets.items():
    manifest = json.load(open(f'{path}/manifest.json'))
    specs = read_jsonl(f'{path}/itemspecs.jsonl')
    views = read_jsonl(f'{path}/promptviews.jsonl')
    
    m_items = manifest['counts']['itemspecs']
    m_views = manifest['counts']['promptviews']
    ok = (m_items == len(specs) and m_views == len(views))
    status = 'PASS' if ok else 'FAIL'
    print(f'{status} {name}: manifest says {m_items} items / {m_views} views, '
          f'files have {len(specs)} / {len(views)}')
"
```

---

## 10. Cross-Suite Metric Consistency [MUST]

Verify all eval scripts produce the same metric format:

- [ ] All scripts write `summary.json` with keys: `UAI_irr`, `UAI_plaus`, `TAR_irr`, `TAR_plaus`, `Disc_delta`, `MAE_control`, `Acc10_control`, `parse_rate`
- [ ] All scripts use `epsilon = 3.0` for UAI exclusion
- [ ] All scripts use `denom = a - y_control` (not `a - y_star`)

```bash
# After running metrics unification (alignment plan 1.2), verify by checking:
rg "anchor_val - y_star" scripts/eval/run_*.py
# Should return NO matches.

rg "EPSILON\|epsilon" scripts/eval/unified_metrics.py
# Should show EPSILON = 3.0
```

---

## Summary: Quick Pass/Fail

| Check | Status | Blocking? |
|---|---|---|
| 1. Schema completeness | [ ] | Yes |
| 2. Paired-condition completeness | [ ] | Yes |
| 3. Count balance | [ ] | Yes |
| 4. Gold-answer validity | [ ] | Yes |
| 5. Parser sanity | [ ] | Yes |
| 6. Deterministic generation | [ ] | Yes |
| 7. Condition naming | [ ] | Yes |
| 8. Anonymization | [ ] | Yes (pre-submission) |
| 9. Manifest consistency | [ ] | Yes |
| 10. Metric consistency | [ ] | Yes |

**All 10 checks must pass before launching full model experiments.**

---

*End of checklist.*
