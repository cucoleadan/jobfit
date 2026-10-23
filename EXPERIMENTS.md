# jobfit: experiment log

Every run that produced a number: what ran, on what data, with what settings, and the result. Paper tables and figures are built from this file and the files it points to. `DECISIONS.md` says *why*; this file says *what happened*.

Conventions:
- Check each run's log line `encoded ...; train N ex` against the dataset version it was meant to use: Kaggle can serve the previous dataset version for a few minutes after an upload reports `ready`.
- Dates are UTC days. Costs are list price unless noted.
- Raw outputs live in `data/` (gitignored: contains posting text). Publishable copies of metrics go in `results/<run>/metrics.json` (no posting or CV text).
- Each training run records: model id, dataset version, seed, all CLI args, GPU hours, per-epoch validation and final test metrics.

## Data runs

| Date | Run | Input | Output | Cost | Notes |
|---|---|---|---|---|---|
| 2026-10-22 | token count | one real CV + 200 postings | CV ~1.7k tokens, pair median ~2.55k, p90 ~3.25k, max < 4,096 | $0 | 3 tokenizers within ~5% after whitespace collapse |
| 2026-10-22 | pilots 1-6 | 20 CVs × 4 pairs each | positive share 1% → 34% | ~$0.70 | see DECISIONS.md, "pilot" entries |
| 2026-10-22 | full run | 2,000 CVs × 4 pairs | 1,973 CVs, 7,891 labelled pairs; verdict skip 5,581 / apply 1,411 / stretch 899; seniority above only 325 | CVs $2.59, labels $10.64, total spend to date $14.33 | `data/full_run.sh`; first pass stopped at 4,382 labels when the 1 h API token expired (401s), resumed after adding token refresh. ~2.76k input / 434 output tokens per pair |
| 2026-10-22 | dataset v1 | 7,891 pairs | split by CV: train 6,315 / val 788 / test 788; 99% of requirements (71,681/72,305) matched to a posting line | $0 | `scripts/build_dataset.py` |
| 2026-10-22 | dataset v2 | v1 + 244 new CVs (976 standard pairs) + 3,833 targeted pairs (`over`, `stretch`) | 2,217 CVs, 12,699 pairs; train 10,171 / val 1,270 / test 1,258 (v1 split frozen); seniority `above` 325 → 1,033, verdict `stretch` 899 → 1,913 | CVs $0.34, labels $6.37; project total $21.03 | `data/v2_run.sh`; 6 CVs and 1 label lost to API errors |

## Browser speed

| Date | Model | Precision | Hardware | 2,048 tok | 3,072 tok | 4,096 tok |
|---|---|---|---|---|---|---|
| 2026-10-22 | ModernBERT-base | fp16 | RTX 3060 6 GB, Chrome 155, Windows, WebGPU | 0.57 s | 0.95 s | 1.43 s |
| 2026-10-22 | ModernBERT-base | q4f16 | same | 0.56 s | 0.93 s | 1.39 s |
| 2026-10-22 | mmBERT-base | fp16 | same | failed: "memory access out of bounds" (256k vocabulary) | | |

## Training runs

| Date | Run | Model | Data | Epochs | GPU | Time | Result |
|---|---|---|---|---|---|---|---|
| 2026-10-22 | timing-base | ModernBERT-base | 475 real CV × posting pairs, **random labels** | 1 | Kaggle 2×T4 | 1.3 min/epoch (0.9M tokens), 1.16 s/step, ~3 min total | fits at 4,096 tokens; metrics meaningless (random labels); weights discarded |
| 2026-10-22 | ettin32 | Ettin-32M (31.9M params) | v1, lr 1e-4, seed 0 | 3 | Kaggle 2×T4 | ~4-5 min/epoch, 18 min total | test: verdict acc 0.789 / macro-F1 0.536; location 0.829 / 0.824; seniority 0.748 / 0.508; must-haves met 0.769 / 0.759; requirement lines 0.879 / 0.621. ECE ≤ 0.04 after temperature scaling. Val still rising at epoch 3 → likely undertrained. `results/ettin32/` |
| 2026-10-22 | ettin68 | Ettin-68M (68.2M params) | v1, lr 1e-4, seed 0 | 3 | Kaggle 2×T4 | ~10-12 min/epoch, 38 min total | test: verdict acc 0.808 / macro-F1 0.589; location 0.846 / 0.839; seniority 0.751 / 0.511; must-haves met 0.787 / 0.779; requirement lines 0.891 / 0.661. ECE ≤ 0.041. Beats 32M on every head; seniority flat. Val verdict F1 still rising at epoch 3. `results/ettin68/` |
| 2026-10-22 | mbbase | ModernBERT-base (149M params) | v1, lr 5e-5, seed 0 | 3 | Kaggle 2×T4 | ~18-21 min/epoch, 68 min total | test: verdict acc 0.797 / macro-F1 0.550; location 0.826 / 0.819; seniority 0.728 / 0.495; must-haves met 0.770 / 0.763; requirement lines 0.882 / 0.630. ECE ≤ 0.053. **Below Ettin-68M on every head**: slower start (val location F1 0.48 after epoch 1), likely undertrained at 3 epochs / lr 5e-5. `results/mbbase/` |
| 2026-10-22 | mblarge | ModernBERT-large (395M) | v1, lr 3e-5, 8,192 tokens/GPU/step | - | Kaggle 2×T4 | failed after 3 min | CUDA out of memory on the first step (14.5 GB used of 15 GB). Rerun on v2 as `mblarge-v2` with 4,096 tokens/step × accum 8 (same effective batch): OOM again at the first optimizer step (fp32 AdamW state ≈ 6.3 GB/GPU). Third try: `--optim adamw8bit` (bitsandbytes) + `expandable_segments`; first launch got the previous dataset version (stale train.py, 'unrecognized arguments'), relaunched a few minutes later. Fourth launch trained at 1.12 s/step (~61 min/epoch) but stopped at step ~680 with DDP "parameters that were not used in producing loss": one-example batches sometimes have no requirement lines, so the line heads got no gradient. Fixed with zero-weight terms in the loss (numerically identical); relaunched |
| 2026-10-22 | ettin68-v1e8 (launched as `ettin68-v2e8`) | Ettin-68M | **v1** (not v2: Kaggle served the previous dataset version; log shows 6,315 train / 788 test), lr 1e-4, seed 0 | 8 | Kaggle 2×T4 | ~13 min/epoch, 105 min total | test (v1 test, 788 ex): verdict acc 0.802 / macro-F1 0.626; location 0.883 / 0.875; seniority 0.735 / 0.576; must-haves met 0.792 / 0.778; requirement lines 0.894 / 0.681. vs `ettin68` (same data, 3 epochs): verdict +0.037, seniority +0.065, location +0.036 → **an epochs-only ablation**. Val verdict F1 plateaus from epoch 5 (best epoch 7). Found 2026-10-23 when the v1 re-score matched this run's test numbers exactly. `results/ettin68-v1e8/` |
| 2026-10-22 | mblarge-v2 | ModernBERT-large (395M params) | **v2**, lr 3e-5, 4,096 tokens/step × accum 8, 8-bit AdamW, seed 0 | 3 | Kaggle 2×T4 | ~63-72 min/epoch, 221 min total | test (v2 test, 1,258 ex): verdict acc 0.818 / macro-F1 0.674; location 0.907 / 0.906; seniority 0.785 / 0.708; must-haves met 0.816 / 0.803; requirement lines 0.900 / 0.698. ECE ≤ 0.045. **Best on every head**, still rising at epoch 3 (val verdict 0.56 → 0.64 → 0.69). Kaggle status ERROR after the test: rank 1 hit the 10-min NCCL timeout while rank 0 evaluated; metrics and weights were already saved. Fixed with a 1 h timeout. `results/mblarge-v2/` |
| 2026-10-23 | e68-v2-e3-lr5e5 | Ettin-68M | v2, lr 5e-5, seed 0 | 3 | Kaggle 2×T4 | 53 min training | val (best epoch 3) F1: verdict 0.647, seniority 0.677, location 0.894, must 0.830, lines 0.651. no test_v1 (launched before train.py reported it). Train 10,171 ex (v2 confirmed in log). `results/e68-v2-e3-lr5e5/` |
| 2026-10-23 | e68-v2-e3-lr1e4 | Ettin-68M | v2, lr 1e-4, seed 0 | 3 | Kaggle 2×T4 | 55 min training | val (best epoch 3) F1: verdict 0.661, seniority 0.700, location 0.893, must 0.849, lines 0.679. test_v1 F1: verdict 0.633, seniority 0.577, location 0.877, must 0.816, lines 0.677. Train 10,171 ex (v2 confirmed in log). `results/e68-v2-e3-lr1e4/` |
| 2026-10-23 | e68-v2-e3-lr2e4 | Ettin-68M | v2, lr 2e-4, seed 0 | 3 | Kaggle 2×T4 | 57 min training | val (best epoch 3) F1: verdict 0.665, seniority 0.682, location 0.899, must 0.848, lines 0.692. test_v1 F1: verdict 0.634, seniority 0.585, location 0.866, must 0.812, lines 0.684. Train 10,171 ex (v2 confirmed in log). `results/e68-v2-e3-lr2e4/` |
| 2026-10-23 | e68-v2-e6-s0 | Ettin-68M | v2, lr 1e-4, seed 0 | 6 | Kaggle 2×T4 | 102 min training | val (best epoch 4) F1: verdict 0.689, seniority 0.640, location 0.896, must 0.852, lines 0.699. test_v1 F1: verdict 0.638, seniority 0.577, location 0.878, must 0.825, lines 0.695. Train 10,171 ex (v2 confirmed in log). `results/e68-v2-e6-s0/` |
| 2026-10-23 | e68-v2-e6-s1 | Ettin-68M | v2, lr 1e-4, seed 1 | 6 | Kaggle 2×T4 | 106 min training | val (best epoch 5) F1: verdict 0.682, seniority 0.697, location 0.906, must 0.855, lines 0.688. test_v1 F1: verdict 0.671, seniority 0.626, location 0.876, must 0.818, lines 0.689. Train 10,171 ex (v2 confirmed in log). Seed 0 vs 1 differ by 0.033 verdict and 0.049 seniority on test_v1: seed noise is as large as most effects measured so far. `results/e68-v2-e6-s1/` |

### Findings so far (shared v1 test set, 788 examples, macro-F1; one seed unless noted)

| Model | Data | Epochs | Verdict | Seniority | Location | Must-haves | Req. lines |
|---|---|---|---|---|---|---|---|
| Ettin-68M | v1 | 3 | 0.589 | 0.511 | 0.839 | 0.779 | 0.661 |
| Ettin-68M | v1 | 8 | 0.626 | 0.576 | 0.875 | 0.778 | 0.681 |
| Ettin-68M | v2 | 3 | 0.633 | 0.577 | 0.877 | 0.816 | 0.677 |
| Ettin-68M | v2 | 6 | 0.638 | 0.577 | 0.878 | 0.825 | 0.695 |
| ModernBERT-large | v2 | 3 | 0.659 | 0.663 | 0.885 | 0.806 | 0.695 |

- **Data (clean ablation, rows 1 vs 3):** targeted v2 pairs at fixed model/epochs/LR: verdict +0.044, seniority +0.066, must-haves +0.037.
- **Epochs:** on v1, 3 → 8 epochs helped (+0.037 verdict); on v2, 3 → 6 adds +0.005 verdict (within expected seed noise; seeds pending).
- **LR (Ettin-68M, v2, 3 epochs, val verdict F1):** 5e-5 0.647, 1e-4 0.661, 2e-4 0.665: flat between 1e-4 and 2e-4; 1e-4 kept.
- **Size:** ModernBERT-large leads mainly on seniority (+0.086 over Ettin-68M v2).
- ModernBERT-large v2 `test_v1` re-scored locally on CPU with `scripts/eval_v1.py` (±0.002 vs GPU, checked on Ettin-32M).

## Planned

- Size sweep on the full dataset: Ettin-32M, Ettin-68M, ModernBERT-base, ModernBERT-large; then Ettin-1B or a generative LLM depending on the curve (PLAN.md step 6c).
- Human evaluation set: 2-3 annotators label the same ~150 pairs; report inter-annotator agreement (Cohen's / Fleiss' κ) and model-vs-human, teacher-vs-human.
