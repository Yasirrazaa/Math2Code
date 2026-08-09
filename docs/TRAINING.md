# Training runbook (GPU box)

Weeks 1–3 run on CPU for **$0**. This runbook covers the paid GPU work
(weeks 4–8): exact commands + hard budget gates. Read `PLAN.md` §Budget first.

## 0. Budget gates (hard)

| Stage | Cost | Gate to proceed |
|-------|------|-----------------|
| SFT warmup (free T4, Colab/Kaggle) | $0 | val loss < 0.5 |
| 1.5B GRPO burn-in (free T4) | $0 | mean reward > 0.5 on held-out 100 prompts; else **stop & re-scope** |
| 7B GRPO main (spot 3090/4090) | ~$25–45 | burn-in passed; ≤ 12h or $45, whichever first |
| API baselines (DeepSeek + GPT-4o-mini) | ~$6–11 | — |
| E2B final eval | $0 (hobby credits) | — |
| **Total** | **~$35–60** | hard cap **$100** |

## 1. Spin up the box (Vast.ai or RunPod)

Instance: RTX 3090/4090 (24 GB), CUDA 12.x image, ~20 GB disk, spot pricing
($0.30–0.60/h). Then:

```bash
git clone https://github.com/Yasirrazaa/Math2Code.git && cd Math2Code
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12 && uv venv && uv pip install -e ".[dev,train]"

# train.json (51MB) is gitignored — loader auto-extracts the committed zip:
python -c "from math2code.data.competition import load_competition_train; print(len(load_competition_train()), 'rows')"
python scripts/make_splits.py          # frozen split, seed 42, deterministic
make eval-gold                         # sanity: 1.0000 (397/397) BEFORE any training
```

## 2. SFT warmup (free T4)

```bash
python -m math2code.model.train model=Qwen/Qwen2.5-Math-1.5B \
  format=sft train_file=./data/split/train.jsonl \
  output_dir=./runs/sft-warmup-1.5b \
  num_train_epochs=1 per_device_train_batch_size=4 gradient_accumulation_steps=8
```

Gate: val loss < 0.5. Push to HF for the GRPO step:

```bash
python -m huggingface_hub.cli.upload_folder \
  --repo_id your-handle/math2code-sft-1.5b --folder-path runs/sft-warmup-1.5b
```

## 3. 1.5B GRPO burn-in (free T4) — configs/grpo.yaml

```bash
python -m math2code.model.grpo --config-dir configs --config-name grpo \
  model_name_or_path=your-handle/math2code-sft-1.5b \
  output_dir=./runs/grpo-burnin-1.5b \
  max_steps=300
```

Gate: **mean reward > 0.5** on the 100 held-out prompts. The config ships with
the burn-in profile (800 prompts, lr 5e-7, 2s/512MB rollouts, batch 8×4×8).
Verify reward decomposition lands in `results/grpo-burnin-1.5b/*.json` and log
a Weave/W&B run.

## 4. 7B GRPO main run (spot GPU, ~$25–45)

```bash
python -m math2code.model.grpo --config-dir configs --config-name grpo \
  model_name_or_path=Qwen/Qwen2.5-Math-7B \
  output_dir=./runs/grpo-main-7b \
  max_steps=400 num_gpus=1 \
  lr=3e-7
```

Hard stop at 12h or $45. Checkpoint every 100 steps → HF Hub. Use the 1.5B
run's reward curve to pick the step (not the wallclock).

## 5. Final eval + submission

```bash
# harness on the local pool (identical to the gold run)
python -m math2code.evaluation.runner --split data/split/test.json \
  --model hf:your-handle/math2code-grpo-7b --save-code results/code_grpo7b.csv
# E2B-sandboxed variant (no local pool state — the plan's final-eval protocol)
E2B_API_KEY=... python -m math2code.evaluation.e2e_eval \
  --split data/split/test.json --code results/code_grpo7b.csv
# publish dataset + model card
HF_TOKEN=... python scripts/publish_hf.py --split data/split --repo your-handle/math2code
```

## 6. What to record per run (for the README table)

- `per_problem_accuracy` + 95% CI (runner prints it; `scripts/analyze_results.py --by equation_type` gives the slice breakdown)
- wall time + $ cost (instance rate × hours)
- reward curve stats, best step, checkpoint id
- any divergence/hardware notes (keep it honest)

## 7. Failure playbook

| Symptom | Action |
|---------|--------|
| Pool workers crash under concurrency | Self-healing pool retries once; raise `memory_mb` if sympy OOMs on dsolve rows |
| Reward flat / NaN | Check `model/rewards.py` reward decomposition; drop lr; restore SFT checkpoint |
| RLVR diverges by step 200 | Stop, restore SFT checkpoint, halve lr, increase entropy coef |
| Parse-baseline beats the model | The calculus slice is the target — check `analyze_results.py`; if non-calculus regressed, SFT on augmented hard negatives (adversarial mutator) |
| Budget exceeds gate | Stop immediately; burn-in already proved the reward signal; report honestly |
