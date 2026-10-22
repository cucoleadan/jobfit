# jobfit: experiment log

Every run that produced a number: what ran, on what data, with what settings, and the result. Paper tables and figures are built from this file and the files it points to. `DECISIONS.md` says *why*; this file says *what happened*.

Conventions:
- Dates are UTC days. Costs are list price unless noted.
- Raw outputs live in `data/` (gitignored: contains posting text). Publishable copies of metrics go in `results/<run>/metrics.json` (no posting or CV text).
- Each training run records: model id, dataset version, seed, all CLI args, GPU hours, per-epoch validation and final test metrics.

## Data runs

| Date | Run | Input | Output | Cost | Notes |
|---|---|---|---|---|---|
| 2026-10-22 | token count | one real CV + 200 postings | CV ~1.7k tokens, pair median ~2.55k, p90 ~3.25k, max < 4,096 | $0 | 3 tokenizers within ~5% after whitespace collapse |
| 2026-10-22 | pilots 1-6 | 20 CVs × 4 pairs each | positive share 1% → 34% | ~$0.70 | see DECISIONS.md, "pilot" entries |
| 2026-10-22 | full run | 2,000 CVs × 4 pairs | 1,973 CVs, 7,891 pairs; labelling in progress | CVs $2.59; labels est. ~$10.5 | `data/full_run.sh`; first pass stopped at 4,382 labels when the 1 h API token expired (401s), resumed after adding token refresh |

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

## Planned

- Size sweep on the full dataset: Ettin-32M, Ettin-68M, ModernBERT-base, ModernBERT-large; then Ettin-1B or a generative LLM depending on the curve (PLAN.md step 6c).
- Human evaluation set: 2-3 annotators label the same ~150 pairs; report inter-annotator agreement (Cohen's / Fleiss' κ) and model-vs-human, teacher-vs-human.
