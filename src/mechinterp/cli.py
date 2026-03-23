"""CLI entrypoint for mechanistic interpretability pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .int_vocab import build_int_vocab
from .logit_scoring import score_integer_distribution, char_to_token_span, wrap_chat
from .metrics import compute_all, js_divergence
from .logit_lens import logit_lens_sweep
from .patching import ResidualPatcher
from .plotting import plot_cae_vs_layer, plot_top_heads, plot_logit_lens
from .tables import generate_imprint_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_items(
    promptviews_path: Path, itemspecs_path: Path,
    max_items: int | None, seed: int,
) -> list[dict]:
    """Load and stratify-sample items. Returns list of item dicts with prompt views."""
    pv_by_id: dict[str, dict[str, dict]] = {}
    with open(promptviews_path, encoding="utf-8") as f:
        for line in f:
            pv = json.loads(line.strip())
            pv_by_id.setdefault(pv["item_id"], {})[pv["condition"]] = pv

    specs: dict[str, dict] = {}
    with open(itemspecs_path, encoding="utf-8") as f:
        for line in f:
            spec = json.loads(line.strip())
            specs[spec["item_id"]] = spec

    items = []
    for item_id, views in pv_by_id.items():
        if "control" not in views:
            continue
        if "low_anchor" not in views or "high_anchor" not in views:
            continue
        spec = specs.get(item_id, {})
        items.append({
            "item_id": item_id,
            "suite": views["control"]["suite"],
            "domain": views["control"]["domain"],
            "y_star": spec.get("y_star", spec.get("theta")),
            "anchor_gap": spec.get("anchors", {}).get("gap", 60),
            "control": views["control"],
            "low_anchor": views["low_anchor"],
            "high_anchor": views["high_anchor"],
        })

    if max_items and max_items < len(items):
        rng = np.random.RandomState(seed)
        by_group: dict[str, list] = {}
        for it in items:
            key = f"{it['suite']}_{it['domain']}"
            by_group.setdefault(key, []).append(it)

        per_group = max(1, max_items // len(by_group))
        sampled = []
        for group_items in by_group.values():
            rng.shuffle(group_items)
            sampled.extend(group_items[:per_group])

        if len(sampled) < max_items:
            remaining = [it for it in items if it not in sampled]
            rng.shuffle(remaining)
            sampled.extend(remaining[:max_items - len(sampled)])

        items = sampled[:max_items]

    log.info("Loaded %d items (suites: %s)", len(items),
             {s: sum(1 for it in items if it["suite"] == s)
              for s in sorted(set(it["suite"] for it in items))})
    return items


def _run_imprint(
    model: Any, tokenizer: Any, int_vocab: dict[int, list[int]],
    items: list[dict], out_dir: Path,
) -> None:
    """Compute distributional anchor imprint for all items."""
    csv_path = out_dir / "imprint_metrics.csv"
    fieldnames = [
        "item_id", "suite", "domain", "condition",
        "ev_ctrl", "ev_low", "ev_high",
        "delta_ev_low", "delta_ev_high",
        "kl_low", "kl_high", "sym_kl_low", "sym_kl_high",
        "js_low", "js_high", "tvd_low", "tvd_high",
        "w1_low", "w1_high", "nai",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, item in enumerate(items):
            log.info("[imprint %d/%d] %s", i + 1, len(items), item["item_id"])
            try:
                p_ctrl, _ = score_integer_distribution(
                    model, tokenizer, item["control"]["prompt_text"], int_vocab
                )
                p_low, _ = score_integer_distribution(
                    model, tokenizer, item["low_anchor"]["prompt_text"], int_vocab
                )
                p_high, _ = score_integer_distribution(
                    model, tokenizer, item["high_anchor"]["prompt_text"], int_vocab
                )
                m = compute_all(p_ctrl, p_low, p_high, item.get("anchor_gap", 60))
                row = {
                    "item_id": item["item_id"],
                    "suite": item["suite"],
                    "domain": item["domain"],
                    **m,
                }
                writer.writerow(row)
                f.flush()
            except Exception as e:
                log.warning("Failed on %s: %s", item["item_id"], e)

    log.info("Imprint metrics -> %s", csv_path)


def _run_patching(
    model: Any, tokenizer: Any, int_vocab: dict[int, list[int]],
    items: list[dict], out_dir: Path,
    do_heads: bool, top_k_layers: int,
) -> None:
    """Layer sweep and optional head sweep patching."""
    patcher = ResidualPatcher(model, int_vocab)

    layer_csv = out_dir / "layerwise_effect.csv"
    layer_fields = ["item_id", "suite", "domain", "condition", "layer",
                    "cae", "js_patched", "js_base"]
    head_csv = out_dir / "top_heads.csv"
    head_fields = ["item_id", "suite", "domain", "condition",
                   "layer", "head", "effect_size", "js_patched"]

    with open(layer_csv, "w", newline="", encoding="utf-8") as lf, \
         open(head_csv, "w", newline="", encoding="utf-8") as hf:
        lw = csv.DictWriter(lf, fieldnames=layer_fields)
        lw.writeheader()
        hw = csv.DictWriter(hf, fieldnames=head_fields)
        hw.writeheader()

        for i, item in enumerate(items):
            log.info("[patching %d/%d] %s", i + 1, len(items), item["item_id"])
            for cond_key in ("low_anchor", "high_anchor"):
                try:
                    pv = item[cond_key]
                    ctrl_text = wrap_chat(item["control"]["prompt_text"], tokenizer)
                    anch_text = wrap_chat(pv["prompt_text"], tokenizer)

                    ctrl_ids = tokenizer(ctrl_text, return_tensors="pt",
                                         add_special_tokens=False)["input_ids"].to(model.device)
                    anch_ids = tokenizer(anch_text, return_tensors="pt",
                                         add_special_tokens=False)["input_ids"].to(model.device)

                    p_ctrl, _ = score_integer_distribution(
                        model, tokenizer, item["control"]["prompt_text"], int_vocab
                    )
                    p_anch, _ = score_integer_distribution(
                        model, tokenizer, pv["prompt_text"], int_vocab
                    )

                    token_positions = None
                    if pv.get("anchor_span"):
                        try:
                            cs, ce = pv["anchor_span"]
                            ts, te, _ = char_to_token_span(
                                pv["prompt_text"], cs, ce, tokenizer
                            )
                            token_positions = list(range(ts, te))
                        except Exception:
                            token_positions = None

                    results = patcher.layer_sweep(
                        ctrl_ids, anch_ids, p_ctrl, p_anch, token_positions
                    )
                    for r in results:
                        r.update({
                            "item_id": item["item_id"],
                            "suite": item["suite"],
                            "domain": item["domain"],
                            "condition": cond_key,
                        })
                        lw.writerow(r)
                    lf.flush()

                    if do_heads:
                        cae_by_layer = {r["layer"]: r["cae"] for r in results}
                        top_layers = sorted(cae_by_layer, key=cae_by_layer.get,
                                            reverse=True)[:top_k_layers]
                        head_results = patcher.head_sweep(
                            ctrl_ids, anch_ids, p_ctrl, p_anch,
                            top_layers, token_positions,
                        )
                        for hr in head_results:
                            hr.update({
                                "item_id": item["item_id"],
                                "suite": item["suite"],
                                "domain": item["domain"],
                                "condition": cond_key,
                            })
                            hw.writerow(hr)
                        hf.flush()

                except Exception as e:
                    log.warning("Patching failed on %s/%s: %s",
                                item["item_id"], cond_key, e)

    log.info("Layer sweep -> %s", layer_csv)
    if do_heads:
        log.info("Head sweep -> %s", head_csv)


def _run_logit_lens(
    model: Any, tokenizer: Any, int_vocab: dict[int, list[int]],
    items: list[dict], out_dir: Path,
) -> None:
    """Logit lens sweep for all items."""
    csv_path = out_dir / "logit_lens.csv"
    fields = ["item_id", "suite", "domain", "condition", "layer", "js_divergence"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, item in enumerate(items):
            log.info("[logit_lens %d/%d] %s", i + 1, len(items), item["item_id"])
            try:
                ctrl_text = wrap_chat(item["control"]["prompt_text"], tokenizer)
                ctrl_ids = tokenizer(ctrl_text, return_tensors="pt",
                                     add_special_tokens=False)["input_ids"].to(model.device)
                ctrl_dists = logit_lens_sweep(model, ctrl_ids, int_vocab)

                for cond_key in ("low_anchor", "high_anchor"):
                    anch_text = wrap_chat(item[cond_key]["prompt_text"], tokenizer)
                    anch_ids = tokenizer(anch_text, return_tensors="pt",
                                         add_special_tokens=False)["input_ids"].to(model.device)
                    anch_dists = logit_lens_sweep(model, anch_ids, int_vocab)

                    for layer_idx, (pd_c, pd_a) in enumerate(zip(ctrl_dists, anch_dists)):
                        js = js_divergence(pd_a, pd_c)
                        writer.writerow({
                            "item_id": item["item_id"],
                            "suite": item["suite"],
                            "domain": item["domain"],
                            "condition": cond_key,
                            "layer": layer_idx,
                            "js_divergence": f"{js:.6f}",
                        })
                f.flush()
            except Exception as e:
                log.warning("Logit lens failed on %s: %s", item["item_id"], e)

    log.info("Logit lens -> %s", csv_path)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate sub-command."""
    parser = argparse.ArgumentParser(description="Mechanistic interpretability pipeline")
    sub = parser.add_subparsers(dest="command", help="sub-command")

    run_p = sub.add_parser("run", help="Run mechinterp analysis")
    run_p.add_argument("--model_id", required=True)
    run_p.add_argument("--promptviews", type=Path, required=True)
    run_p.add_argument("--itemspecs", type=Path, required=True)
    run_p.add_argument("--max_items", type=int, default=None)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--out_dir", type=Path, default=Path("results/mechinterp"))
    run_p.add_argument("--device", type=str, default="auto")
    run_p.add_argument("--dtype", type=str, default="bfloat16")
    run_p.add_argument("--do_patching", action="store_true")
    run_p.add_argument("--do_heads", action="store_true")
    run_p.add_argument("--top_k_layers", type=int, default=8)

    agg_p = sub.add_parser("aggregate", help="Generate figures/tables from results")
    agg_p.add_argument("--auto_discover", type=Path)

    args = parser.parse_args()

    if args.command is None:
        old_args = args
        args = argparse.Namespace(
            command="run",
            model_id=getattr(old_args, "model_id", None),
            promptviews=getattr(old_args, "promptviews", None),
            itemspecs=getattr(old_args, "itemspecs", None),
            max_items=getattr(old_args, "max_items", None),
            seed=getattr(old_args, "seed", 42),
            out_dir=getattr(old_args, "out_dir", Path("results/mechinterp")),
            device=getattr(old_args, "device", "auto"),
            dtype=getattr(old_args, "dtype", "bfloat16"),
            do_patching=getattr(old_args, "do_patching", False),
            do_heads=getattr(old_args, "do_heads", False),
            top_k_layers=getattr(old_args, "top_k_layers", 8),
        )
        if args.model_id is None:
            parser.print_help()
            sys.exit(1)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "aggregate":
        _cmd_aggregate(args)


def _cmd_run(args: argparse.Namespace) -> None:
    """Load model, run imprint / logit-lens / patching, and write artifacts."""
    model_short = args.model_id.split("/")[-1]
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"{model_short}_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

    provenance = {
        "model_id": args.model_id,
        "git_commit": _git_hash(),
        "seed": args.seed,
        "max_items": args.max_items,
        "do_patching": args.do_patching,
        "do_heads": args.do_heads,
        "top_k_layers": args.top_k_layers,
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "dtype": args.dtype,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    log.info("Loading model %s ...", args.model_id)
    device_map = args.device if args.device != "auto" else "auto"
    torch_dtype = getattr(torch, args.dtype, "auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype,
        device_map=device_map, trust_remote_code=True,
    )
    model.eval()
    log.info("Model loaded: %s on %s", args.model_id, model.device)

    int_vocab = build_int_vocab(tokenizer)
    items = _load_items(args.promptviews, args.itemspecs, args.max_items, args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    _run_imprint(model, tokenizer, int_vocab, items, out_dir)
    log.info("Imprint done in %.1fs", time.time() - t0)

    _run_logit_lens(model, tokenizer, int_vocab, items, out_dir)
    log.info("Logit lens done in %.1fs", time.time() - t0)

    if args.do_patching:
        _run_patching(model, tokenizer, int_vocab, items, out_dir,
                      args.do_heads, args.top_k_layers)
        log.info("Patching done in %.1fs", time.time() - t0)

    _generate_artifacts(out_dir, model_short)
    log.info("All mechinterp done in %.1fs. Output: %s", time.time() - t0, out_dir)


def _generate_artifacts(out_dir: Path, model_name: str) -> None:
    """Generate figures and tables from CSVs."""
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"

    if (out_dir / "layerwise_effect.csv").exists():
        plot_cae_vs_layer(out_dir / "layerwise_effect.csv",
                          fig_dir / "cae_vs_layer.pdf", model_name)
    if (out_dir / "top_heads.csv").exists():
        plot_top_heads(out_dir / "top_heads.csv", fig_dir / "top_heads.pdf")
    if (out_dir / "logit_lens.csv").exists():
        plot_logit_lens(out_dir / "logit_lens.csv",
                        fig_dir / "logit_lens.pdf", model_name)
    if (out_dir / "imprint_metrics.csv").exists():
        generate_imprint_table(out_dir / "imprint_metrics.csv",
                               tab_dir / "mechinterp_main.tex")


def _cmd_aggregate(args: argparse.Namespace) -> None:
    """Discover result dirs and regenerate all figures/tables."""
    base = args.auto_discover
    if not base or not base.exists():
        log.error("Directory not found: %s", base)
        return
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and (sub / "provenance.json").exists():
            prov = json.loads((sub / "provenance.json").read_text())
            model_name = prov.get("model_id", "").split("/")[-1]
            log.info("Regenerating artifacts for %s", sub.name)
            _generate_artifacts(sub, model_name)


if __name__ == "__main__":
    main()
