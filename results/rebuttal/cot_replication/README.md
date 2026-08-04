# CoT replication run (camera-ready)

Independent third run of the **Llama-8B x External x {baseline, cot}** cell,
launched to resolve the disagreement between:

- `results/revision/mitigation_headroom/` : UAI_pls 0.336 -> 0.338 (max_tokens=512, batch_size=64)
- `results/rebuttal/cot_extended/`        : UAI_pls 0.394 -> 0.366 (max_tokens=768, batch_size=32)

This run uses the **same configuration as `cot_extended`** (max_tokens=768,
batch_size=32, vLLM, greedy / temperature 0):

```bash
python -m anchorbench.runners.rebuttal_cot \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --backend vllm --tensor_parallel_size 1 --gpu_memory_utilization 0.85 \
    --suites external --strategies baseline cot \
    --batch_size 32 --max_tokens 768 --out_dir <this dir>
```

## Result

| run                          | UAI_pls base | UAI_pls CoT | delta   | Disc_d base | Disc_d CoT |
|------------------------------|--------------|-------------|---------|-------------|------------|
| mitigation_headroom          | 0.3358       | 0.3378      | +0.0020 | 0.2952      | 0.3095     |
| cot_extended                 | 0.3943       | 0.3663      | -0.0280 | 0.3109      | 0.3096     |
| **this replication**         | 0.3183       | 0.3465      | +0.0282 | 0.2665      | 0.2474     |

## Conclusion

The two original runs render **byte-identical prompts, gold answers and anchor
values** (verified record-by-record), and both decode greedily. The divergence
is run-to-run variation of batched GPU inference, which is not bitwise
reproducible at temperature 0 because batch composition changes kernel
reduction order.

Between `cot_extended` and this replication, 13.1% of baseline generations and
7.9% of parsed baseline answers differ. Across the three runs UAI_pls (baseline)
spans 0.32-0.39 and Disc_delta spans 0.27-0.31.

**The sign of the CoT effect for this single cell is not resolved by one run.**
Aggregate conclusions in Appendix "Reasoning-allowed (CoT) evaluation" are
unaffected: they rest on cells with effects up to -0.34, an order of magnitude
larger, and UAI_pls stays positive under CoT in all three runs.

Documented in the paper at `COLM_camera_ready/sections/appendix.tex`,
footnote `fn:cot-variance`.
