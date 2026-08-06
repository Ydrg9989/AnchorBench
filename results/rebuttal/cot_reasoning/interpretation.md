# CoT (reasoning-allowed) vs. baseline analysis
## Context
Does anchoring persist when the model is allowed to
reason before answering, rather than being forced to output a
single integer on the last line? The existing
`mitigation_headroom` runs already include a CoT condition
(*"Think step by step. List the relevant evidence, compute your
estimate from that evidence only, then provide your final numeric
answer on the last line."*) for Qwen-7B and Llama-8B on External
and RAG. The numbers below come from those existing runs.
## Per-cell comparison
### external / Llama-8B
- baseline:  UAI_irr=0.041, UAI_pls=0.336, Disc=0.295, MAE_ctrl=5.5, Acc10=0.829
- + reasoning: UAI_irr=0.028, UAI_pls=0.338, Disc=0.309, MAE_ctrl=6.1, Acc10=0.828
- delta:    UAI_pls=+0.002, Disc=+0.014, MAE_ctrl=+0.6, Acc10=-0.001
### external / Qwen-7B
- baseline:  UAI_irr=0.013, UAI_pls=0.271, Disc=0.258, MAE_ctrl=7.7, Acc10=0.725
- + reasoning: UAI_irr=0.007, UAI_pls=0.150, Disc=0.143, MAE_ctrl=1.1, Acc10=0.964
- delta:    UAI_pls=-0.121, Disc=-0.115, MAE_ctrl=-6.6, Acc10=+0.239
### rag / Llama-8B
- baseline:  UAI_irr=-0.042, UAI_pls=0.120, Disc=0.162, MAE_ctrl=4.8, Acc10=0.843
- + reasoning: UAI_irr=0.009, UAI_pls=0.092, Disc=0.083, MAE_ctrl=4.5, Acc10=0.859
- delta:    UAI_pls=-0.028, Disc=-0.079, MAE_ctrl=-0.4, Acc10=+0.016
### rag / Qwen-7B
- baseline:  UAI_irr=-0.004, UAI_pls=0.197, Disc=0.201, MAE_ctrl=9.8, Acc10=0.639
- + reasoning: UAI_irr=-0.002, UAI_pls=0.029, Disc=0.031, MAE_ctrl=1.3, Acc10=0.969
- delta:    UAI_pls=-0.168, Disc=-0.170, MAE_ctrl=-8.5, Acc10=+0.331

## Headline
- Across 4 (model, suite) cells with a CoT comparison:
  - 3/4 cells show *reduced* UAI_pls under reasoning
    (delta UAI_pls < -0.01).
  - 0/4 cells show *increased* UAI_pls.
  - 1/4 cells *nearly eliminate* plausible-anchor
    susceptibility (baseline UAI_pls > 0.1 drops to <= 0.05).
  - mean delta UAI_pls = -0.079.
- mean delta Acc10 (control accuracy) = +0.146; reasoning generally *also* improves accuracy.

## Summary
Reasoning-allowed prompting *attenuates* anchoring but does not
eliminate it. On Qwen-7B (External) UAI_pls drops from 0.27 to 0.15
(-45%) while control accuracy rises from 73% to 96%; on RAG the same
model drops from 0.20 to 0.03. Llama-8B on External, however, is
almost unchanged (0.34 -> 0.34). The effect is therefore not a pure
artifact of the constrained-output instruction. Caveat: this is two
models on two pathways; Tier B1 extends to 5 models on External +
RAG + History.
