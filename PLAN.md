# Math2Code TIR — Master Blueprint v3 (final)

> **What this is:** the complete, grounded plan for turning Math2Code into a portfolio-grade project — a LaTeX→executable-Python translation system trained with Tool-Integrated Reasoning (TIR) and GRPO, with honest, reproducible evaluation.
>
> **What changed since v1/v2:** measurement-first ordering (eval harness + frozen test set before any training), the bootcamp competition data is now available locally (`data/train.json`: 26,846 verified samples with ground-truth SymPy + test cases), RL is built on TRL/OpenRLHF (no custom rollout engine), rewards are deterministic rule-based (no LLM judges), sandboxing is local-first (nsjail with fallbacks) with E2B reserved for final eval, and the timeline includes a small-model burn-in gate before the expensive run.

---

## 0. Non-negotiable principles

1. **Measurement first.** The frozen test set, the eval harness, and baseline numbers (zero-shot external models) exist *before* any RL run. No training step is started without a known baseline to beat.
2. **Stand on battle-tested frameworks.** No custom rollout loops, no hand-rolled RL math. TRL (`environment_factory`) → OpenRLHF (`--agent_func_path`) → verl (`restricted_python`) as the scale ladder.
3. **Deterministic rewards only.** Every reward term is a rule (string match, execution result, `sympy.count_ops`). No LLM-as-judge inside the RL loop.
4. **Cheap sandbox at train time, strong sandbox at eval time.** Local process/nsjail sandbox for rollouts (~10–50 ms/execution); E2B or Docker for the canonical benchmark.
5. **The repo must be honestly green.** Broken build, red CI, and a leaking eval are blocking bugs, not background noise.

---

## 1. Data reality & strategy

### 1.1 What exists (verified locally)

| Asset | Records | Ground truth | Use |
|---|---|---|---|
| `data/train.json` (competition) | 26,846 | `sympy_exp` + `test_cases[].output` (5 each) | Primary training + GRPO reward target |
| `data/public_test_new_no_sol_no_out.json` | 1,004 | inputs only (no outputs) | Final eval (with hidden answers) |
| `data/final/synthetic_data_final.json` | 5,013 | code only, **no test cases** | Clean/deprecate; secondary |
| `data/results_v14 (1).csv` | 1,004 | — | Your 0.75 baseline submission |

**Competition data facts (audited):**
- `train.json`: all `synthetic: true`, all `output_type: real`; 5 test cases per row, **none missing expected outputs**; complexity 2 (15k) → 5 (1.9k); 8 domains, ~13 equation types (derivative, differential, integration, summation, diophantine, rational, exponential…); **~4k duplicate LaTeX expressions** (dedup required).
- `public_test`: 906 real + **98 complex** output types; harder than train (complexity 3–5 dominates); **~536 items are out-of-distribution families absent from train** — 500 `augmented_equation` plus `logrithmic`, `fractional`. This is a generalization probe and explains part of the 0.75 ceiling.
- Your 0.75 submission is well-formed (1,004 ids × 5 outputs, 0 parse failures); 120/490 complex-expected outputs produce complex values (complex handling mostly works); 31 non-finite values score 0.

### 1.2 Strategy

1. **Primary dataset = competition train.json** (deduplicated → ~23k unique LaTeX). Its `sympy_exp` field is the symbolic ground truth — the "oracle" is already embedded.
2. **Re-sampling for GRPO:** do *not* train against the fixed 5 test cases (memorization). At rollout time, re-derive 3–5 *fresh random numeric inputs* per prompt by substituting into `sympy_exp` (seeded by task_id). Expected outputs are computed from the ground-truth AST — the training-time oracle (see §2).
3. **Clean or deprecate `data/final/synthetic_data_final.json`:** its solutions are chatty (```python fences + trailing prose) and it has no test cases. Migrate it through the same cleaning pipeline (strip fences/prose, extract code, attach re-sampled test cases) or drop it in favor of competition data.
4. **Extension (Phase 2+, optional):** the code-first `MathOntologyGenerator` (SymPy AST → `sp.latex()` prompt; `max_depth`/`operator_set` curriculum) becomes a *supplementary* generator to cover underrepresented families (complex outputs, ODEs/differential, matrices, piecewise) — validated by the same oracle before entering the pool. Never trust LLM-generated data without sympy verification.
5. **Versioning:** publish the canonical dataset to **HF Hub** (`datasets.push_to_hub`) — visible to recruiters, single `load_dataset` call in train/eval code. Keep `data/train.json.zip` (7.8 MB) and small files in git; **gitignore the extracted 51 MB `train.json`**.

---

## 2. Verification oracle & sandboxing

### 2.1 The oracle (deterministic ground truth)

For every candidate sample (competition row, re-sampled rollout input, or AST-generated pair):

1. **Syntactic gate:** code must compile (`SyntaxError`/`ImportError` → reject).
2. **Numeric gate (Monte Carlo):** substitute ≥50 random values (drawn from ℂ to catch branch-cut errors; integer-valued for diophantine) into both the candidate expression and the ground truth. Compare `cmath.isclose(rel_tol=1e-5, abs_tol=1e-8)` (real: `math.isclose`; complex: both parts).
3. **Pole avoidance:** if any denominator evaluates to `|x| < 1e-9`, discard the point and resample (avoid division-by-zero noise).
4. **Symbolic gate (stronger, when cheap):** `sp.simplify(candidate - ground_truth) == 0` — catches rare mismatches Monte Carlo can miss on continuous expressions.

The oracle is used at **data build time** (validate/tag samples) and at **RL reward time** (fresh random inputs, no caching possible → no memorization).

### 2.2 The training sandbox (local)

**Primary: `nsjail`** — `seccomp-bpf` syscall filtering + `rlimit` CPU/memory caps + network namespace isolation.
- Profile: no network, 2.0 s CPU cap, 512 MB memory, run as `nobody` inside a minimal chroot rootfs (Debian snapshot or busybox + Python + sympy).
- Throughput: ~10–50 ms/snippet → 10k executions in minutes.
- **Deployment friction (plan for it):** nsjail needs root *or* unprivileged user namespaces enabled, and a chroot rootfs. It will not run on macOS and is painful in Docker/CI without `--cap-add SYS_ADMIN` + adjusted AppArmor.

**Fallbacks (design the sandbox as an interface, not a single tool):**
- `python-seccomp` + `setrlimit` subprocess (no root needed, works on Linux).
- Docker-per-batch: `docker run --network=none --memory=512m --cpus=1 --pids-limit=64` (strong, ~batch overhead).

**Final eval only:** E2B (cloud) or Docker for citation-grade isolation in the benchmark.

---

## 3. TIR protocol & environment loop

### 3.1 Protocol (deterministic parse)

```text
<think>
[Reasoning about the problem and next step...]
</think>
<execute>
[Valid Python code using sympy; defines `calculate(...)` and prints the value]
</execute>
<observation>
[Stdout/Stderr from sandbox execution — injected by the environment, never by the model]
</observation>
```

### 3.2 Loop semantics

1. Generate until `</execute>` (vLLM stop string) → extract code by regex.
2. Execute in local sandbox on the row's re-sampled inputs → capture stdout/stderr.
3. Inject as `<observation>` → resume generation.
4. Terminate on `<final_answer>` tag **or** max turns (≥1 required execute, cap at 5) **or** context limit.

### 3.3 Framework (no custom rollout engine)

| Framework | Route | When |
|---|---|---|
| **TRL 1.9.x** | `GRPOTrainer(reward_funcs=[...], args=GRPOConfig(use_vllm=True, num_generations=8, scale_rewards="batch"))` — sandbox execution in the reward (already verified against 1.9.2; see `model/grpo.py` + `tests/test_grpo.py`) | Start here (single node) |
| **OpenRLHF 0.10.x** | `--agent_func_path` multi-turn agent mode (Ray + vLLM) | If multi-node scaling needed |
| **verl 0.8.x** | built-in `restricted_python` code-exec env, native step-level rewards | If step-level advantage needed (see §4.3) |

**Status (Week 5, verified):** TRL 1.9.x's `environment_factory`/`BaseEnvironment` API was removed — 1.9.x environments are tool-calling based and don't fit base math models, so the shipped entrypoint is reward-funcs-only GRPO: the reward executes each completion's `<execute>` code in the sandbox against per-rollout resampled inputs, and `task_id` flows from the dataset column into reward kwargs. If observation-token masking is later required, it lives on the OpenRLHF/verl rung of the ladder, not TRL. Cap `max_completion_length` accounting for context growth (Phase 4 serves at `--max-model-len 8192`).

---

## 4. Reward design (dense, rule-based, anti-hacking)

One reward function per completion, sum of rule-based terms (TRL computes group-normalized advantage per completion — so what we get is a **dense trajectory reward**, not per-token/per-step advantages; if true step-level credit is required later, that's verl's territory, not TRL).

```python
import re, math, sympy as sp

def reward_func(completions, **kwargs):
    rewards = []
    for completion, prompt, row in zip(completions, kwargs["prompts"], kwargs["rows"]):
        code = extract_code(completion)                    # regex over <execute>...</execute>
        obs  = extract_last_observation(completion)        # last <observation>...</observation>
        r = 0.0

        # R_terminal (+2.0): Monte Carlo oracle on fresh random inputs
        if final_value := extract_final_value(completion):
            r += 2.0 if oracle_matches(final_value, row, seed=hash(row["task_id"])) else 0.0

        # R_exec (+1.0 / -0.25): structured result from sandbox (exit code), not string sniffing
        if obs is not None:
            r += +1.0 if obs.exit_code == 0 else -0.25

        # R_tool (+0.1): reward attempting execution (counteracts tool-avoidance collapse)
        if "<execute>" in completion:
            r += 0.1

        # R_meta (+0.2): rule-based traceback attention
        m = re.search(r"NameError: name '(\w+)' is not defined", obs or "")
        if m and re.search(rf"\b{m.group(1)}\s*=\s*|\bsymbols\(.*{m.group(1)}", code or ""):
            r += 0.2

        # R_complexity (−0.01 × Δcount_ops): only on correct final expressions
        if final_value is not None and oracle_matches(final_value, row):
            r -= 0.01 * max(0, sp.count_ops(final_expr) - sp.count_ops(row["ground_truth_expr"]))
        rewards.append(r)
    return rewards
```

### 4.1 Why these choices (and what they prevent)

| Term | Prevents | Notes |
|---|---|---|
| **R_tool** | Tool-avoidance collapse (model stops executing to dodge error penalties, then hallucinates) | Errors should be mildly negative (−0.25), **not** −1.0 |
| **R_meta** | None (it's shaping) | String-match on the traceback token + the next execute block; weak weight |
| **R_complexity** | Verbose/expanded non-simplified answers | `sympy.count_ops`; only on *correct* answers so it can't be gamed by returning trivial wrong expressions |
| **R_terminal** | — | Fresh random inputs per rollout → no input memorization |

### 4.2 SFT warmup first

Before RL, SFT the base model on TIR-format trajectories (competition rows with `<think>/<execute>/<observation>` synthesized by executing the gold solution, plus a handful of *deliberately erroneous* trajectories with corrected continuations from the adversarial mutator). RL then optimizes an already-format-stable policy — this is what prevents format collapse during GRPO.

### 4.3 Step-level credit (honest framing)

Standard GRPO gives one group-normalized advantage per completion. A **dense trajectory reward** already provides credit for good intermediate steps (successful executes accumulate reward regardless of final correctness). True per-step advantage — only if needed and measured to help — moves the loop to **verl** (native step rewards). Do not hand-roll advantage math.

---

## 5. Training plan

### 5.1 Base models

| Stage | Model | Why |
|---|---|---|
| SFT warmup | `AI-MO/NuminaMath-7B-TIR` (your 2024 base, TIR-native) | Already speaks tool-integrated reasoning |
| GRPO burn-in (gate) | `Qwen/Qwen2.5-Math-1.5B` | Cheap iteration on loop stability/rewards before spending GPU money |
| GRPO main | 7B class (Qwen2.5-Math-7B or continue from SFT'd NuminaMath) | Headline model |

### 5.2 Compute & cost budget

- **Burn-in gate:** 1.5B GRPO, 8 rollouts × 1k prompts, ~1024 completion tokens — hours on one 24 GB GPU. **Gate: rewards non-zero, loop stable, no tool-avoidance collapse, eval pass@1 does not regress vs SFT.** Do not proceed to 7B until the gate passes.
- **Main run:** 7B LoRA GRPO on 4×A100 (or 1×80 GB H100), ~1–3 days/run, budget 3–5 runs. Cost ≈ $1–3k at current spot rates — decide the cap *before* launching.
- Always: `bf16`, `gradient_checkpointing`, small `per_device_train_batch_size` + accumulation; vLLM rollouts on a separate device (`vllm_device`).

### 5.3 Logging

W&B: KL divergence, rollout entropy, per-reward-term means (watch R_tool stay high), group reward distribution, and **eval pass@1 on the frozen test set after every K steps** (via a small async eval worker).

---

## 6. Evaluation harness & benchmark protocol

### 6.1 The metric (replicates the competition scoring)

- Per test case: parse submitted output (stringified number, possibly complex, e.g. `'1.23e-4'`, `'-10.096475+88.331647j'`) → numeric compare with `math.isclose(rel_tol=1e-6, abs_tol=1e-9)` (complex: both parts).
- Aggregate: **per-case accuracy** and **per-problem accuracy** (all 5 cases correct), plus syntax-failure rate, token cost, latency.
- Report **pass@1 (greedy)** and **pass@k (k=32–64, unbiased estimator)** with **bootstrap 95% CIs**.

### 6.2 The frozen test set

- 300–500 rows held out from competition data, **stratified by domain × complexity**, never touched by training/HP search; stored with SHA-256 + split indices in git.
- Contamination check: zero `latex_expression` overlap with training data.
- Sanity check: ground-truth gold code must score 1.0 on the harness (validates metric implementation against `train.json`).

### 6.3 Baseline benchmark matrix (the README table, finally real)

| Model | pass@1 | pass@k | per-case | notes |
|---|---|---|---|---|
| GPT-4o-mini (0-shot) | | | | API cost tracked |
| Claude Sonnet 4 (0-shot) | | | | |
| DeepSeek-R1 / V3 (0-shot) | | | | |
| Qwen2.5-Math-7B-Instruct (0-shot) | | | | |
| DeepSeek-Math-7B (0-shot) | | | | |
| NuminaMath-7B-TIR (0-shot) | | | | your 2024 base |
| **Your 0.75 submission (results_v14)** | | | | record baseline |
| Math2Code SFT (this project) | | | | |
| **Math2Code SFT + GRPO (this project)** | | | | headline |

Same prompt template, same sandbox, same test set for every row. Results auto-rendered into `README.md` from `results/latest.json`; every run logged to W&B.

### 6.4 Error analysis (the portfolio story)

Breakdown by `equation_type` (esp. the OOD families: `augmented_equation`, `logrithmic`, `fractional`; the 98 complex-output problems; `differential`/ODEs). This shows *what RL fixes and what it doesn't*.

---

## 7. Production topology & observability

### 7.1 Docker Compose (modern GPU syntax, pinned versions)

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.26.0        # pin current stable; document choice
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2                       # MUST match tensor-parallel-size
              capabilities: [gpu]
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model /models/math2code-grpo-final
      --tensor-parallel-size 2
      --max-model-len 8192
      --served-model-name math2code
    ports: ["8000:8000"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]

  api:
    build: .
    depends_on: [vllm]
    environment:
      - VLLM_BASE_URL=http://vllm:8000/v1
    ports: ["8080:8080"]

  app:                                          # Gradio, streams <think>/<observation>
    build: .
    command: python src/serve/app.py
    depends_on: [api]
    ports: ["8501:8501"]
```

Notes: no `runtime: nvidia` (obsolete); `--enable-auto-tool-choice` is irrelevant to the custom XML protocol (drop it); healthchecks on every service; `.env` for secrets, never baked into images; non-root user; multi-stage Dockerfile (Phase 0 fixes the currently-broken build).

### 7.2 Observability — W&B Weave + OpenTelemetry

- **Weave OTLP endpoint** ingests OTLP traces (standard W&B doesn't ingest OTel natively).
- Instrument: vLLM generation (vLLM has **native OTel support** via `VLLM_OTEL_*` — use it rather than wrapping the OpenAI SDK), nsjail execution, AST parsing, each loop turn.
- Trace the full trajectory `Prompt → Think → Execute → Observation → Final Answer` correlated with latency and token cost. Grafana/Tempo as the self-hosted alternative.

---

## 8. Execution timeline (8 weeks, full-time focus)

| Week | Milestone | Definition of done |
|---|---|---|
| **1** | Foundation | `pip install -e .`, CI (lint+type+test) green; data contract fixed (`python_code`→`solution` bug); frozen test set (stratified, checksummed); eval harness v1 with competition metric passing the gold-code sanity check (score 1.0 on train rows) |
| **2** | Data & sandbox | Competition data loader + dedup (~4k dupes removed) + HF Hub publish; local sandbox interface (nsjail + fallback) passing 10k-execution smoke in <5 min; oracle (syntactic + MC numeric + pole avoidance) unit-tested |
| **3** | Baselines | 0-shot benchmarks (GPT-4o-mini, Claude, DeepSeek, Qwen2.5-Math, NuminaMath) + your 0.75 submission on the frozen test set → **first real table with CIs** |
| **4** | SFT warmup | TIR-format SFT on competition data (subset) + adversarial-mutator hard negatives; eval vs Week-3 table |
| **5** | TIR env loop + rewards | Reward funcs unit-tested on static trajectories; TRL 1.9.x reward-func contract verified (sandbox execution in-reward, `task_id` via dataset column); **1.5B GRPO burn-in gate** (100-step sanity: rewards nonzero, no collapse) |
| **6** | Full GRPO | 7B GRPO main run(s) on 4×A100; pass@1 monitored on frozen test during training |
| **7** | Final eval | Same harness + E2B for production-grade numbers: **RL vs SFT vs all baselines**, pass@1/pass@k, CIs, error analysis by equation_type |
| **8** | Production & publish | Docker stack green, Weave traces streaming, Gradio UI (live `<think>`/`<observation>`), HF model + dataset publish, README results table, model card, docs |

**Gates:** Week 5 (1.5B burn-in) and Week 6 (cost cap) are hard gates — no skipping.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Reward hacking / input memorization | fresh random inputs per rollout from `sympy_exp`; R_complexity only on correct answers; KL guard; frozen-test re-eval with *new* random inputs |
| Tool-avoidance collapse | R_tool shaping, mild error penalty (−0.25 not −1), ≥1 forced execute in warmup SFT |
| TRL API churn (verified: 1.9.x dropped `environment_factory`/`BaseEnvironment`) | shipped entrypoint targets the verified 1.9.2 reward-func API; pin TRL version; fallback ladder to OpenRLHF → verl |
| nsjail deployment friction | sandbox as interface with `python-seccomp` + Docker fallbacks; containerized CI with proper caps |
| OOD test families (500 augmented_equation etc.) | explicit eval slice + error analysis; targeted data extension if RL underperforms there |
| 7B GRPO cost overrun | 1.5B burn-in gate + fixed run budget ($1–3k cap) decided in advance |
| Data contamination (base models trained on similar math) | 0-shot baselines before training; private/re-sampled test inputs; note in model card |
| Repo regression | CI (lint/type/test/build) mandatory on every PR; eval smoke (50 items) in CI |

---

## 10. Definition of done (repo-level)

- Fresh clone → `make setup && make lint && make typecheck && make test` green; `docker compose build` green.
- `python -m evaluation.evaluate --config configs/eval/base.yaml` reproduces `results/latest.json` (any model spec).
- `README.md` shows the full benchmark table (this project's SFT + GRPO vs. 0-shot external models) with CIs — no "Pending" rows.
- Dataset + model published to HF Hub; W&B project public; demo on HF Spaces.
- A 2-page write-up: problem, protocol, results, error analysis, 2–3 annotated RL trajectories (before/after).

---

## Appendix A — Toolchain (verified, Aug 2025)

`trl 1.9.2` (GRPOTrainer, GRPOConfig, environment_factory experimental, use_vllm) · `verl 0.8.0` (restricted_python) · `OpenRLHF 0.10.4` · `math-verify 0.9.0` (parse/verify exact-match for eval) · `vllm 0.26.0` (native OTel, LoRA, serving) · `e2b-code-interpreter 2.9.0` · `evalplus 0.3.1` (pass@k math) · `lm-eval 0.4.12` · `inspect-ai 0.3.253` (sandbox-native eval alternative) · `transformers 5.x` · `peft 0.20.x` · `datasets 5.x` · `wandb 0.28.x` · `sympy` (count_ops, physics.units optional)

## Appendix B — Local data files (git hygiene)

- **Commit:** `data/train.json.zip` (7.8 MB), `data/public_test_new_no_sol_no_out.json`, `data/public_test_new_sample_submission.csv`, `data/results_v14 (1).csv`, split indices + test-set checksums.
- **Gitignore:** extracted `data/train.json` (51 MB), `data/raw/`, `data/processed/`, `outputs/`, caches.
- **Canonical dataset:** HF Hub (`your-user/math2code-competition`), regenerated test-case variants for RL pushed as dataset revisions.
