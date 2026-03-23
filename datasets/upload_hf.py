# Upload AnchorBench HF-release promptviews. Generate first from repo root:
#   python scripts/export_public_promptviews.py --data_dir datasets --suites external history icl rag tool --size core --hf
# Then from datasets/: python upload_hf.py
# (If you see "No files have been modified since last commit", the file content is already on the Hub.)
from pathlib import Path
from huggingface_hub import HfApi

repo_id = "Yiderigun/LLM_anchoring"
api = HfApi()

for folder in Path(".").glob("anchorbench_*_core"):
    file = folder / "promptviews_public.jsonl"
    if not file.exists():
        print(f"Skip {folder}: no promptviews_public.jsonl (run export with --hf first)")
        continue

    suite = folder.name.replace("anchorbench_", "").replace("_core", "")
    target = f"{suite}_core/promptviews.jsonl"

    print(f"Uploading {file} -> {target}")

    result = api.upload_file(
        path_or_fileobj=str(file),
        path_in_repo=target,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("Uploaded:", result)

print("All done.")