# Experiments — Final Evaluation Matrix

Full-scale experiments for the AnchorBench v1 paper.

## Structure

```
experiments/
  configs/             # YAML configs per experiment group
    model_sweep.yaml   # model × decoding grid
    mitigation.yaml    # mitigation strategy ablations
    mechanistic.yaml   # activation-patching configs
  run_experiments.py   # orchestrator (local HF + API models)
  outputs/             # raw JSONL outputs (gitignored; see manifests)
```

## Planned experiment matrix

| Axis              | Values                                                  |
|-------------------|---------------------------------------------------------|
| Models            | Llama-3.2-{1B,3B}, Llama-3.1-8B, Qwen-3-4B, GPT-4o-mini, Claude-3.5-haiku |
| Decoding          | greedy, sampling (T=0.7, n=20)                          |
| Mitigations       | none, EFR, CoT-debias, AALC (α∈{0.5,1.0,1.5})          |
| Mechanistic       | activation patching (layers × positions)                |

## Running

```bash
python experiments/run_experiments.py \
  --config experiments/configs/model_sweep.yaml \
  --dataset benchmark/data/anchorbench_v1.jsonl
```
