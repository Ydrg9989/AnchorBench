"""Data loading, filtering, subsampling, and permutation reconstruction."""

import ast
import random
from collections import defaultdict

from datasets import load_dataset


def load_regime_a():
    """Load tum-nlp dataset, filter to Anchoring, parse metric_params.

    Returns list of dicts with keys:
        idx, scenario, control, treatment, x_1, k, flip_treatment
    """
    ds = load_dataset("tum-nlp/cognitive-biases-in-llms")["train"]
    anchoring = ds.filter(lambda x: x["bias"] == "Anchoring")

    rows = []
    for i in range(len(anchoring)):
        row = anchoring[i]
        mp = ast.literal_eval(row["metric_params"])
        rows.append({
            "idx": i,
            "scenario": row["scenario"],
            "control": row["control"],
            "treatment": row["treatment"],
            "x_1": mp["x_1"],
            "k": mp["k"],
            "flip_treatment": mp["flip_treatment"],
        })
    return rows


def load_regime_b():
    """Load jecht dataset, reconstruct permutation blocks grouped by id.

    Returns dict: scenario_id -> {
        students: [str],
        n_students: int,
        permutations: [[int]],   # each is a permutation of student indices
        n_permutations: int,
    }
    """
    ds = load_dataset(
        "jecht/cognitive_bias",
        data_files={"anchoring_bias": "anchoring/students.csv"},
    )["anchoring_bias"]

    id_to_prompts = defaultdict(list)
    for i in range(len(ds)):
        id_to_prompts[ds[i]["id"]].append(ds[i]["prompts"])

    scenarios = {}
    for scenario_id, prompts_list in id_to_prompts.items():
        unique_students = list(dict.fromkeys(prompts_list))
        n_students = len(unique_students)
        student_to_idx = {s: i for i, s in enumerate(unique_students)}

        n_perms = len(prompts_list) // n_students
        permutations = []
        for p in range(n_perms):
            block = prompts_list[p * n_students : (p + 1) * n_students]
            permutations.append([student_to_idx[s] for s in block])

        scenarios[scenario_id] = {
            "students": unique_students,
            "n_students": n_students,
            "permutations": permutations,
            "n_permutations": n_perms,
        }

    return scenarios


def subsample_regime_a(rows, n_per_bin=5, seed=42):
    """Stratified subsample by x_1 bin (1-9)."""
    rng = random.Random(seed)
    bins = defaultdict(list)
    for row in rows:
        bins[row["x_1"]].append(row)

    sampled = []
    for x1 in sorted(bins.keys()):
        pool = bins[x1]
        k = min(n_per_bin, len(pool))
        sampled.extend(rng.sample(pool, k))
    return sampled


def subsample_regime_b(scenarios, k_scenarios=5, p_perms=5, seed=42):
    """Pick K scenario IDs at random, cap permutations at P."""
    rng = random.Random(seed)
    all_ids = sorted(scenarios.keys())
    selected = rng.sample(all_ids, min(k_scenarios, len(all_ids)))

    sub = {}
    for sid in selected:
        sc = scenarios[sid]
        perms = sc["permutations"][:p_perms]
        sub[sid] = {**sc, "permutations": perms, "n_permutations": len(perms)}
    return sub
