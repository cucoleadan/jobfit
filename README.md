# jobfit

A small model that tells a job seeker whether a job is worth applying to, given their CV, and why: apply / stretch / skip, location eligibility, seniority fit and the requirements the CV does not show. It runs in the browser (WebGPU), so the CV never leaves the person's device.

Status: training data is being generated; models not trained yet.

- [PLAN.md](PLAN.md): goal, measurements, pipeline, paper track
- [DECISIONS.md](DECISIONS.md): every design choice, why, and what was rejected
- [EXPERIMENTS.md](EXPERIMENTS.md): every run and number

## How it works

1. **Teacher labels.** DeepSeek V4.1 Flash (open weights, MIT) reads a CV and a job posting and judges facts: each requirement met / partial / not met, location eligibility, the posting's required years and level, which CV roles are relevant. Python applies the rubric thresholds, so every label follows the same published rules (`prompts/label_pair.md`).
2. **Synthetic CVs around a target job.** Each training CV is written for a fictional person whose career is sampled from real postings, anchored on a target job with a controlled level of fit (`prompts/synth_cv.md`, `scripts/pipeline.py`).
3. **Students.** ModernBERT / Ettin encoders (32M-395M) learn every head in one pass over "CV [SEP] job" at 4,096 tokens (`train/train.py`), trained on free Kaggle GPUs.
4. **Browser.** ONNX q4f16 on WebGPU: ~1 s per job for the 149M model on a mid-range laptop GPU (`web/speedtest.html`).

## Running it

```
cp local.example.json local.json      # your OpenAI-compatible endpoint for DeepSeek V4.1 Flash, paths, Kaggle user
python scripts/pipeline.py postings   # needs a postings sqlite db (not distributed)
python scripts/pipeline.py cvs --n 20 && python scripts/pipeline.py pairs && python scripts/pipeline.py label
python scripts/build_dataset.py
python scripts/kaggle_run.py upload && python scripts/kaggle_run.py push --model jhu-clsp/ettin-encoder-32m --name ettin32
```

Scraped job postings are not redistributed; code, prompts, synthetic CVs, labels (planned), weights and evaluation results are.
