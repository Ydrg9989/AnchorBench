# Analysis — Final Figures & Tables

Post-hoc analysis pipeline: metrics aggregation, figure generation, and
LaTeX-ready tables for the AnchorBench v1 paper.

## Structure

```
analysis/
  compute_metrics.py    # aggregate raw outputs → summary statistics
  generate_figures.py   # produce camera-ready figures
  generate_tables.py    # produce LaTeX table sources
  figures/              # output PNGs / PDFs
  tables/               # output .tex snippets
```

## Key outputs

| Artifact                      | Description                              |
|-------------------------------|------------------------------------------|
| `figures/nai_by_model.pdf`    | NAI (Normalized Anchoring Index) by model |
| `figures/dose_response.pdf`   | Anchor magnitude → response shift        |
| `figures/mitigation_ablation.pdf` | Mitigation strategy comparison       |
| `figures/patching_heatmap.pdf`| Activation-patching layer × position     |
| `tables/main_results.tex`    | Table 1: main NAI results                |
| `tables/mitigation.tex`      | Table 2: mitigation effectiveness        |

## Usage

```bash
python analysis/compute_metrics.py \
  --inputs experiments/outputs/ \
  --out    analysis/

python analysis/generate_figures.py --data analysis/metrics.json
python analysis/generate_tables.py  --data analysis/metrics.json
```
