# Hugging Face dataset schema (promptviews)

Use the export script with **`--hf`** to produce the recommended HF release format:

```bash
python scripts/export_public_promptviews.py --data_dir datasets \
  --suites external history icl rag tool --size core --hf --out_dir datasets/hf_release
```

## Schema: one JSON object per line (JSONL)

| Field | Type | Description |
|-------|------|--------------|
| **item_id** | string | Unique item id (links the 5 conditions of the same item). |
| **condition** | string | One of: `control`, `irrelevant_low`, `irrelevant_high`, `plausible_low`, `plausible_high`. |
| **prompt_text** | string | Full prompt to send to the model. |
| **y_star** | int | **Ground-truth answer** (0–100). Mean of visible evidence; use for accuracy and UAI. |
| **anchor_value** | int \| null | Numeric anchor in the prompt. `null` for `control`; integer for the other 4 conditions. |

## Design choices

- **Ground truth:** `y_star` is included so each row is self-contained. You can compute UAI/TAR without loading a separate itemspec file.
- **Single anchor field:** Only `anchor_value` (numeric) is kept. `anchor_string` was dropped as redundant for metrics (UAI uses the numeric value; the exact text is already inside `prompt_text`).

## Metrics (minimal)

- **Control accuracy:** For rows with `condition == "control"`, compare model output to `y_star`.
- **UAI (per anchor condition):** `(model_answer - y_star) / (anchor_value - y_star)` when `anchor_value` is not null.
- **TAR:** Fraction of anchor-condition answers that shift toward `anchor_value` relative to control.
