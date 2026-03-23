# Is validation necessary?

## Short answer

- **For releasing data to Hugging Face:** No. Consumers only need the data files (e.g. minimal promptviews with `prompt_text` + `item_id` + `condition`). They do not run our validators.
- **For us (benchmark producers):** Yes. Validation is a safety net before we tag a release or regenerate data. It catches generator bugs and ensures the artifact we publish is internally consistent.

## What validation does

| Check | Purpose |
|-------|--------|
| ItemSpec schema | θ in [30,70], difficulty easy/hard, anchors present, evidence structure, gold = round(mean(visible evidence)) |
| PromptView schema | Answer-format instruction in prompt, control vs anchor condition consistency |
| Pairing | Every item has 5 conditions; scenario / evidence / question identical across conditions (uses `prompt_components`) |
| No duplicates | Unique item_ids, distinct prompt hashes per item |
| Balance | Domain and difficulty counts match the intended design |
| Manifest | File counts match manifest |

## Who needs it?

- **Producers (us):** Run `validate_all.sh` after generation or before freezing. Ensures we don’t ship broken data. Keep the validation code in the repo.
- **Consumers (HF users):** They download promptviews + itemspecs and run the benchmark. They don’t need to run or ship our validators. Validation is unnecessary for them.

## Recommendation

- **Keep** `validate.py` and `validators.py` in the repo as part of the producer workflow.
- **Do not** require consumers to validate; the released dataset is already validated before upload.
- For the HF release, export **minimal** promptviews (e.g. `item_id`, `condition`, `prompt_text` only) so the public artifact is small and simple.
