# Math2Code — Senior ML Engineering Review & Upgrade Blueprint

**Prepared:** August 2025 · **Audience:** Yasir (project owner) · **Scope:** deep code audit + external research + prioritized roadmap to turn the 2024 bootcamp capstone into a portfolio-grade ML engineering project.

---

## 1. Executive summary

The project has a genuinely good idea and a credible skeleton — LaTeX→executable-SymPy translation, sandboxed execution, LoRA SFT, FastAPI+Gradio — but it is currently **not in a shippable state**, and the *claims* in the README outrun the *reality* of the code. A senior ML engineer would first make the repo honestly green (build, CI, tests, real metrics), then invest most of the effort where it produces portfolio signal: **a rigorous, reproducible evaluation harness** (exact-number match + code execution on a held-out test set, benchmarked across models), **data engineering** (test cases, dedup, splits, versioning), and **GRPO / RL-with-verifiable-rewards** training — which is exactly how the public *Math2Code* task family is being solved at the frontier today.

Verified defects found (each with evidence, §2.2):

1. `pip install -e .` **fails** — no package is declared for hatchling (breaks CI *and* the Dockerfile).
2. CI runs only lint + typecheck, and **both fail** (4 ruff errors, 1 mypy src-layout error); pytest is commented out.
3. `train.py` will **crash at runtime** — it reads `sample['python_code']`, the actual dataset key is `solution` (KeyError).
4. `eval.py` does **not** measure functional correctness — it only checks "runs without error". The README's Pass@1 table is empty, so the headline claim is unproven.
5. The eval **test set leaks training data** — it slices the *last 100 rows of the training file*.
6. DVC is decorative — data is committed directly to git (17 MB); `data/raw` and `data/processed` are empty; the curate step was never wired end-to-end.
7. Training dependencies (`torch`, `transformers`, `peft`, `trl`, `hydra`, `wandb`, `datasets`) live only in `requirements.txt` and are **absent from `pyproject.toml`**, which nothing installs.
8. Dataset has **no `test_cases`/expected outputs** — so numeric verification cannot be run against it as-is; 30 duplicate (latex, solution) pairs exist.

None of these are hard to fix. The strategic point: **the task itself is excellent portfolio material**, because "make an LLM produce code that provably computes the right number" is a *verifiable* task — verifiability is what turns a toy fine-tune into a credible story (SFT → RLVR/GRPO → execution-graded benchmarks).

---

## 2. Current state — verified, with evidence

### 2.1 What exists (good bones)

| Layer | What's there | Verdict |
|---|---|---|
| Task | LaTeX expression → runnable SymPy function (`calculate(...)`) | Strong, demo-able, verifiable |
| Data | ~~`data/final/synthetic_data_final.json`~~ — 5,013 rows (`task_id`, `latex_expression`, `solution`) | Legacy capstone rows; removed in the Aug-2026 Phase-0 cleanup (superseded by `data/split/*` + `data/synthetic/*`) |
| Generation | `src/data/generate.py` — Groq llama3-70b + `instructor` + pydantic structured output | Good pattern, tiny scale (3 prompts × 5) |
| Curation | `src/data/curate.py` — E2B sandbox, keeps only error-free snippets | Right idea, only smoke-level |
| Training | `src/model/train.py` — Hydra config, LoRA SFT via TRL `SFTTrainer`, W&B | Decent 2024-era SFT; the notebook used a proper isclose-based judge |
| Eval | `src/evaluation/eval.py` — E2B pass@1 | **Broken metric** (see 2.2) |
| Serve | FastAPI proxy → vLLM (+LoRA) + E2B execution; Gradio UI | Clean microservice idea; code extraction is fragile |
| Ops | docker-compose (vLLM/API/UI), Dockerfile, DVC dir, pre-commit, Makefile | Skeleton only; Docker build is broken; compose hardcodes GPU |
| CI | GitHub Actions: ruff + mypy only | Red today (see 2.2) |

Data quality quick audit (measured):
- 5,013 rows; 3,295 unique LaTeX expressions; 30 duplicate (latex, solution) pairs.
- 99.6% solutions start with a ```python fence; 100% contain a `def`; 99.4% contain `return`; 71% import sympy.
- **No `test_cases` field, no expected outputs** → cannot compute numeric correctness without synthesizing them.
- Solutions are chatty (code block + trailing prose, e.g. *"However, a more idiomatic way to write this…"*) → needs a strict code extractor.

Domain coverage (rough classification of 5,013): fractions ~2.5k, integrals ~0.5k, trig ~0.25k, sqrt ~0.1k, log ~0.05k, derivatives ~32, matrix ~0. Heavily skewed toward fractions/algebra; calculus (derivatives/integrals — the hard, interesting part) is under-represented. This matters because the original bootcamp report says the *competition* dataset emphasized differentiation/integration.

### 2.2 Confirmed defects (reproduced)

```bash
# 1. Package build fails → CI and Docker both break at install time
$ uv pip install --no-deps -e .
  → "At least one file selection option must be defined in the
     tool.hatch.build.targets.wheel table …"   [hatchling can't find a package:
     src/ has 4 sibling subpackages and no top-level `math2code` package]

# 2. Lint/typecheck fail (if CI ever got past install)
$ ruff check src/ tests/
  → 4 errors (I001 import sorting ×3, F401 unused `pytest` import)
$ mypy src/ tests/
  → src/serve/api.py: error: Source file found twice under different module names:
    "api" and "src.serve.api"   [src-layout mypy config issue]

# 3. Training data contract mismatch → train.py KeyError at runtime
$ python3 -c "json.load(open('data/final/synthetic_data_final.json')).keys()"
  → ['task_id', 'latex_expression', 'solution']        # 'python_code' missing!
  → format_instruction() raises KeyError: 'python_code'
```

Dockerfile additionally depends on a `pyproject.toml` that cannot build, and `COPY pyproject.toml .` without `README.md` would break hatchling's `readme` reference even after fixing packaging. `docker-compose.yml` pins `vllm/vllm-openai:latest` (floating tag), hardcodes NVIDIA, passes `E2B_API_KEY` to a container whose build is broken, and mounts `./src` as a volume (dev-only pattern in a "production" compose).

### 2.3 The one historical asset being thrown away

The notebooks (`notebooks/Training and Inference.ipynb`, `notebooks/sympy_Yasir (1).ipynb`) contain the **original, correct evaluation logic** from the bootcamp: extract code → compile → run on `test_cases` → compare with `math.isclose(rel_tol=1e-9)` (with complex-number handling), plus `match_and_prepare_inputs` for parameter-name fuzz. The 2025 `src/` refactor **regressed** this to "executes without error". A senior engineer would rescue that judge logic, harden it, and rebuild the eval harness around it — not leave it in a notebook.

---

## 3. What's worth keeping (don't rewrite these)

1. **The task choice.** LaTeX→code with verifiable outputs is interview-gold: it lets you show *measurement*, not vibes.
2. **The overall pipeline shape** (generate → curate → train → eval → serve) as a mental model.
3. **Structured generation** (`instructor` + pydantic) — good practice, keep.
4. **E2B usage** for final-stage sandboxing — keep for *evaluation*, not for training-time rewards (too slow/costly per sample).
5. **Hydra config + W&B** — keep, extend to eval configs.
6. **LoRA SFT checkpoint path** (`fine_tuned_aimo_lora_model_v3` referenced in notebooks) — your SFT artifact is the natural seed/initialization for GRPO. Preserve it and the exact SFT recipe (data, seeds) so the RL stage is reproducible.
7. **Microservice split** (vLLM engine / API proxy / UI) — the right shape; fix the packaging and make it buildable.

---

## 4. Prior art & inspiration (research notes)

The task family you built is actively researched; citing these in the README/portfolio shows you know the field:

- **Kaggle × Numenta "Math2Code" competition (Dec 2024–Feb 2025)** — the same task: generate Python (`calculate` function) from math problems; scored by **executing code on hidden test cases**. Winning/leaderboard approaches converged on small math models + **RL with execution-based verifiable rewards (GRPO/RLVR)** + best-of-N sampling. This is your strongest external validation that the problem is real and the GRPO direction is the right one. (Your bootcamp competition was private; the public one is the perfect "inspired by" citation.)
- **DeepSeek-R1 (Jan 2025)** — popularized **GRPO with rule-based (verifiable) rewards** for math; your task has a *better* verifiable reward than R1's format/accuracy checks: real code execution.
- **HARP (ByteDance, 2025)** — *Heuristic Abstract Reasoning for Program synthesis*: GRPO over code-execution rewards where the model "reasons in Python" (scratchpad → program → execution feedback). Extremely close to your task; a must-cite.
- **Open-R1 / TRL GRPO recipes** (HF) — the standard open-source RLVR stack: `trl.GRPOTrainer` + `math_verify` rewards, vLLM rollouts.
- **NuminaMath / AI-MO (Progress Prize 1)** — the base model you chose (NuminaMath-7B-TIR) and its **Tool-Integrated Reasoning (TIR)** are part of this lineage; your project can position itself as "TIR for symbolic computation".
- **RLVR (Reinforcement Learning from Verifiable Rewards)** — the general framing: any checkable signal (numeric tolerance, code passes/fails) can be a dense RL reward; no reward model needed.

Tooling verified current (PyPI, Aug 2025): `trl 1.9.2` (GRPOTrainer, GRPOConfig, `use_vllm`, `scale_rewards="batch"`, experimental `environment_factory` for tool-use), `math-verify 0.9.0` (parse/verify exact-match + sympy equivalence), `verl 0.8.0` (byte-level RL framework with built-in `restricted_python` code-execution env), `OpenRLHF 0.10.4`, `vllm 0.26.0`, `e2b-code-interpreter 2.9.0`, `evalplus 0.3.1` (pass@k with Docker sandbox), `lm-eval 0.4.12`, `inspect-ai 0.3.253` (sandbox-native eval framework), `deepeval 4.1.5`, `datasets 5.0.1`.

---

## 5. Roadmap (phased, dependency-ordered)

> Effort estimates assume one engineer, part-time, on a machine with ≥1 CUDA GPU (24 GB VRAM recommended; 7B-class LoRA GRPO is feasible on one RTX 3090/4090).

### Phase 0 — Make it honestly green  *(3–5 days)*

Goal: `make ci` (or the GitHub workflow) passes and `docker compose build` succeeds. Everything after this is credible.

1. **Fix packaging.** Either (a) create a real top-level package `src/math2code/` (recommended) with submodules `data`, `model`, `evaluation`, `serve`, or (b) declare explicit hatchling `packages` and add `__init__.py` files. Add `[tool.hatch.build.targets.wheel] packages = ["src/math2code"]`.
2. **Unify dependency management.** One source of truth in `pyproject.toml` (move torch/transformers/peft/trl/datasets/accelerate/hydra/wandb/vllm-optional into `[project.optional-dependencies] train = [...]`); delete or fold `requirements.txt`. Pin versions (at least major.minor) and add `constraints.txt` for torch+CUDA pins.
3. **Fix the data contract.** Single schema (`MathCodePair` pydantic model, used everywhere): `task_id`, `latex_expression`, `solution` (code only, fenced), `test_cases`, `expected_outputs`. Add a `normalize_data.py` that migrates the existing 5,013 rows to it (strips chatty prose, canonicalizes fences).
4. **Fix lint/typecheck config.** `[tool.mypy]` needs `explicit_package_bases = true` + `mypy_path = "src"` (or run `mypy -p math2code`), fix the 4 ruff issues, enable `ruff format --check` in CI.
5. **Turn on pytest in CI** and fix the 3 existing test files (they mock correctly; make them pass).
6. **Fix Docker.** Multi-stage build (builder installs, runtime copies wheels); non-root user; `COPY README.md` before `pip install -e .`; `.dockerignore`; healthchecks; `deploy.resources` + `profiles: [gpu]` so CPU-only compose works; don't bind-mount `./src` in a "prod" profile (add a `dev` profile for it).
7. **Delete or fix the DVC dir** — decide now (see Phase 2): either wire DVC properly or drop it and version data via HF Hub / git-lfs. A half-configured DVC folder signals the opposite of rigor.

**Definition of done:** fresh checkout → `make setup && make lint && make typecheck && make test` green; `docker compose build` green.

### Phase 1 — Rigorous evaluation (the portfolio's centerpiece) *(1–2 weeks)*

This is the highest-leverage work. A senior ML engineer's credibility lives here.

1. **Build a real test set once, freeze it.**
   - Take the curated 5,013 rows + newly generated data; **dedup** (30 dup pairs today), stratify by difficulty/domain (add a `difficulty` field: algebra / calculus / trig / matrix / complex; easy/medium/hard heuristics: expression length, nesting depth, presence of integrals/derivatives).
   - Hold out **200–500 rows as an immutable test set** (never touched by training or hyperparameter search). Save it as `data/test/` with a checksum; store split indices in git.
2. **Write the metric correctly.** Two complementary scores on every model:
   - **Execution correctness (primary):** parse code → run in sandbox on K=5–10 randomized numeric test cases (inputs drawn from a seeded RNG, expected outputs computed from the ground-truth sympy expression at generation time) → fraction of cases where output matches within tolerance. Handle int/float/complex/Fraction/Expr (use `sympy.N`, `math.isclose(rel_tol=1e-6, abs_tol=1e-9)`, complex → `isclose` on re/im).
   - **Exact number match (secondary, classic):** canonicalize the model's answer string (strip whitespace, unify `1e-6` vs `0.000001`, fractions, complex notation) → exact string equality with the canonical ground truth. Report it alongside execution correctness — this is the "exact number match for a test set" your README already promises.
   - **Also report:** syntax/parse failure rate (why not: 30% of failures are extraction, not reasoning — this is a great error-analysis slide), token count / cost / latency per sample.
3. **Rebuild `src/evaluation/` as a proper harness.** One `evaluate.py` that takes (model spec, test set, config) and returns a JSON report + W&B run. `pass@1` (greedy, `temperature=0`) and `pass@k` (sample k=32/64 @ t=1.0, count unique successes — implement the unbiased pass@k estimator; `evalplus` has the math). Add **bootstrap 95% CIs** (portfolio-grade rigor: "92.1% ± 1.8%").
4. **Local sandbox first, E2B for the final run.** Local: `subprocess` + `resource.setrlimit` (CPU/memory/time) + `seccomp`-filtered Python, or Docker per-batch (`--network=none --pids-limit --memory`). E2B (already paid for by the project) for the canonical, citation-ready numbers.
5. **Benchmark matrix (the README table, finally filled):** the fine-tuned model vs. zero-shot baselines: **GPT-4o-mini, Claude 3.5/4 Sonnet, DeepSeek-V3/R1, Qwen2.5-Math-7B-Instruct, DeepSeek-Math-7B (base), NuminaMath-7B-TIR (your base, zero-shot)** — on *the same* frozen test set, same prompt template, same sandbox. Output one table like the README already promises, plus a per-difficulty breakdown.
6. **Wire eval into CI as a nightly job** on a GPU runner (self-hosted or RunPod/GH Actions larger runner): smoke-eval (50 items) on every PR; full eval nightly; results → W&B + a `results/latest.json` committed or stored.

**Definition of done:** `python -m evaluation.evaluate --config configs/eval/base.yaml` reproduces a JSON report; the README table has real numbers with CIs; the harness is a standalone, documented artifact (this is your interview centerpiece).

### Phase 2 — Data engineering & extension *(1–2 weeks)*

1. **Synthesize `test_cases` for all 5,013 rows.** For each row: parse ground-truth code, extract parameter names from `def calculate(...)`, generate 5–10 random numeric instantiations (seeded), compute expected outputs by executing the *verified ground truth* (which you control — you generated it). Store as `test_cases: [{input: {x: 2.0, y: -1.0}, expected: 0.31}, ...]`. This single step makes every downstream artifact (curation, eval, GRPO reward) possible.
2. **Regenerate with breadth.** The original bootcamp data had ~14 expression types with emphasis on differentiation/integration; the current file is fraction-heavy. Extend generation (`generate.py`): add categories (derivatives, indefinite/definite integrals, limits, series, complex, matrices, piecewise), enforce a per-category quota, and **verify each generated pair by sympy equivalence** (`sympy.simplify(generated - ground_truth) == 0`) — symbolic self-verification is the senior-engineer move that makes LLM-generated data trustworthy without trusting the LLM.
3. **Add public high-quality math data adapted to the format.** MATH (`HuggingFaceH4/MATH-500`, `competition_math`), GSM8K (`openai/gsm8k`), AIME (`HuggingFaceH4/aime_2024`), **NuminaMath-CoT** (`AI-MO/NuminaMath-CoT`, 860k) and **OpenMathInstruct-2** (`nvidia/OpenMathInstruct-2`) can be filtered for expressions whose ground truth is a single numeric/symbolic answer, then converted to LaTeX→code pairs with sympy-generated test cases. (Be careful: models like Qwen2.5-Math were trained on MATH/GSM8K — see contamination risk §10.)
4. **Version the data properly.** Options, pick one and commit to it: (a) **DVC with a real remote** (S3/GCS/R2) + `.dvc` files per dataset — good if you want to show DVC; (b) **HF Hub dataset** (`huggingface_hub` + `datasets.push_to_hub`) — simplest for portfolio, visible to recruiters, and `load_dataset("your-user/math2code")` in train/eval code is a clean story; (c) git-lfs. Recommendation: (b) HF Hub + a small `data/README.md` with the pipeline description. DVC only if you have a remote you'll actually use.
5. **Add dataset cards + provenance.** Per-split provenance (`source: groq-llama3-70b-synthetic`, `source: numina-math-cot`), license notes, contamination warnings. Recruiters *do* read these.

### Phase 3 — GRPO / RLVR training *(2–4 weeks incl. tuning)*

This is the "latest techniques" headline, and your task is unusually well-suited: **code execution gives a dense, verifiable reward with zero reward-model training**.

**Setup (TRL 1.9.x path — recommended first):**
- **Base:** your SFT LoRA (or restart from `Qwen2.5-Math-7B` / continue from NuminaMath-7B-TIR). Small models first (`Qwen2.5-Math-1.5B` / `Qwen3-1.7B`) to iterate cheaply, then scale to 7B.
- **Data:** 1–3k prompts (LaTeX + declared variable list) with their sympy-verified `test_cases`, held-out test set untouched.
- **Rollouts:** TRL `GRPOTrainer` + `GRPOConfig(use_vllm=True, vllm_device="cuda:0"...)` (or the `vllm_mode="server"` path matching your compose), `num_generations=8–16`, `max_completion_length≈1024`, β (KL) ≈ 0.04, lr 1e-6–3e-6.
- **Reward functions (the craft):**
  ```python
  from trl import GRPOTrainer, GRPOConfig

  def format_reward(completions, **kwargs):
      """+0.5 for a valid code block containing `def calculate(`."""
      scores = []
      for c in completions:
          code = extract_code(c)                      # regex over ```python ... ```
          scores.append(0.5 if code and re.search(r"def\s+calculate\s*\(", code) else 0.0)
      return scores

  def execution_reward(completions, **kwargs):
      """Execution-based dense reward: fraction of test cases passing."""
      from math2code.sandbox import run_in_sandbox     # local seccomp/subprocess sandbox
      prompts = kwargs["prompts"]                       # latex expr + vars
      tcs = kwargs["test_cases"]                        # from the dataset row
      scores = []
      for code, cases in zip(completions, tcs):
          code = extract_code(code)
          if code is None:
              scores.append(0.0); continue
          results = run_in_sandbox(code, cases, seed=hash(prompt) % 2**32)  # randomized inputs
          scores.append(sum(results) / len(results))
      return scores

  trainer = GRPOTrainer(
      model="Qwen/Qwen2.5-Math-7B", args=GRPOConfig(...),
      reward_funcs=[format_reward, execution_reward],
      train_dataset=rl_dataset,
  )
  trainer.train()
  ```
- **Anti-reward-hacking (critical, shows senior judgment):** (a) randomize the numeric test inputs *per rollout* (seeded by prompt id) so the model can't memorize inputs; (b) use ≥3–5 test cases per prompt and require all-pass for full reward; (c) keep a small format reward so the model doesn't collapse into garbage that "passes"; (d) watch KL divergence and rollout-entropy in W&B; (e) after training, re-run Phase-1 eval *on the frozen test set with fresh random inputs*.
- **Alternatives if you want to show breadth:** `verl` (`restricted_python` code-exec env is built in; scales to 32B+; used by HARP) or `OpenRLHF`. One is enough for the portfolio; TRL is the most approachable and best-documented.
- **Compute expectations:** 7B LoRA GRPO, 8 rollouts × 1k prompts, ~1024 completion tokens ≈ several hours on one 24 GB GPU per epoch; budget 3–6 runs for tuning. Use `gradient_checkpointing=True`, `bf16`, small `per_device_train_batch_size` with accumulation.
- **Report it like a paper:** SFT baseline vs. GRPO, both at pass@1 and pass@k, with CIs, rollout count, KL, and 2–3 example trajectories (model "thinks in code", gets feedback, corrects — TIR-style). This narrative *is* the portfolio.

### Phase 4 — Ops, docs & presentation *(1 week, parallelizable)*

1. **CI/CD.** GitHub Actions: (a) PR job — lint, typecheck, unit tests, build wheel, docker build; (b) nightly eval job (GPU runner) — smoke eval on PRs, full benchmark nightly → W&B; (c) release job on tag — build + push images to GHCR, publish dataset/model to HF Hub, generate README results table from `results/latest.json`.
2. **Serve hardening.** Code extraction as a tested module (fence-tolerant, handles ```` ```python ```` / bare code / trailing prose); pydantic-settings for env; `E2B_API_KEY` optional with local-sandbox fallback and a clear `X-Sandbox` header; request timeouts, rate limiting, structured logs (JSON), `/healthz` + `/metrics`.
3. **Docs.** Architecture doc (mermaid, from README); **model card** (base, data, training, eval results, limitations, bias); **dataset card**; ADR-style notes for the 3–4 big decisions (why GRPO over DPO; why local sandbox vs E2B at train time; why HF Hub vs DVC); `CONTRIBUTING.md`, issue/PR templates; kill the placeholder badge URL in the README.
4. **Publish.** Model → HF Hub (LoRA adapter + merged), dataset → HF Hub, demo → HF Spaces (E2B or local sandbox behind it), README leaderboard table with your real numbers. A polished repo + Spaces demo + W&B links is the complete package a hiring manager can verify in 10 minutes.

---

## 6. Evaluation protocol spec (the "exact number match + code execution for a test set" deliverable)

**Prompt template (frozen for all models):** one string, consistent across baseline and fine-tuned; document it in `configs/eval/prompt.md`.

**Test set:** N≈300–500, stratified by domain/difficulty, frozen, stored with SHA-256; contamination-checked against the training set (exact `latex_expression` overlap = 0).

**Per-sample protocol:**
1. Sample model output (greedy for pass@1; temperature 1.0 × 32–64 for pass@k).
2. Extract code with the shared extractor (fail → record `extraction_failure`).
3. Execute in sandbox on 5–10 *seeded-random* numeric inputs; compare to ground-truth sympy values.
   - numeric: `math.isclose(got, exp, rel_tol=1e-6, abs_tol=1e-9)`
   - complex: isclose on real & imag parts
   - symbolic: `sympy.simplify(got - exp) == 0` fallback
4. Exact-match (string canonicalization): normalize → compare (report separately).

**Aggregation:** pass@1 (greedy), pass@k (unbiased estimator), syntax-failure rate, token cost, latency; **95% bootstrap CI** on all headline numbers; W&B + JSON artifacts per run; results table auto-rendered into README.

---

## 7. Sandboxing design (defense in depth — show this)

| Layer | Purpose | In your stack |
|---|---|---|
| 1. Static analysis | Reject obviously malicious/unsafe code before execution | AST allowlist (no `os`, `subprocess`, `socket`, `eval/exec`, `open`) |
| 2. Subprocess isolation | OS-level containment | `resource` limits (CPU time, memory, address space), `setrlimit`, run as `nobody` |
| 3. seccomp | Syscall filtering | `python-seccomp` (or `restricted_python` in verl which implements this) |
| 4. Network isolation | Cut exfiltration | `--network=none` (Docker) or `socket` denied via seccomp |
| 5. Full sandbox (final eval only) | Citation-grade isolation | E2B (already used) or Docker per-batch with `--pids-limit --memory=1g --cpus=1` |

Trade-off to document: **local seccomp sandbox at training time** (fast, ~ms/sample, batch-friendly — needed for GRPO throughput) vs. **E2B/Docker at final eval time** (stronger, slower, ~s/sample — fine for 300–500 samples). Saying this explicitly is exactly what a senior engineer sounds like.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Reward hacking** in GRPO (code that "passes" without computing the right thing) | randomized per-rollout inputs, ≥3–5 test cases, format reward, KL guard, frozen-test-set re-eval with fresh inputs |
| **Sandbox escape** from LLM-generated code | defense-in-depth (AST allowlist → rlimits → seccomp → no network → container), never run generated code on the host, documented in README security section |
| **Data contamination** (base models trained on MATH/GSM8K/AIME) | report zero-shot baselines *before* training; use private/synthetic test cases where possible; note in model card |
| **Eval leakage** (current bug: last-100-of-training-file) | fixed frozen split, checksum, contamination check |
| **Cost blowup** (E2B × rollouts × models) | local sandbox at train time; E2B only for canonical eval; cache execution results; cap nightly benchmarks |
| **Scope creep** (GRPO on 7B without iteration budget) | iterate on 1.5B first, then scale; fix compute budget per run in advance |
| **Repo looks green but is red** (today's reality) | Phase 0 first; CI badge must be real before claiming production-readiness |

---

## 9. Suggested sequencing (senior-engineer order)

```
Week 1  Phase 0  — build/CI/tests green, data contract fixed, Docker builds
Week 2  Phase 1  — frozen test set, correct metrics, harness, baselines (GPT-4o-mini, Claude,
                   Qwen2.5-Math, DeepSeek-Math, NuminaMath zero-shot, your SFT) → first real table
Week 3  Phase 2  — test_cases synthesis, dataset extension, versioning (HF Hub)
Week 4  Phase 3a — SFT checkpoint documented; GRPO spike on 1.5B (reward design, sandbox)
Week 5  Phase 3b — GRPO at 7B; pass@1/pass@k vs SFT; error analysis; 2–3 trajectory write-ups
Week 6  Phase 4  — CI/CD nightly benchmarks, docs, HF publish, Spaces demo, README results
```

Gate between phases: the previous phase's "definition of done" is green before starting the next. Phase 1 and 3 are the ones that make you interview-competitive; don't let Phase 4 marketing outrun Phase 1 rigor.

---

## 10. Portfolio presentation guide (how to tell this story)

- **Elevator line:** *"I built a verifiable-math system: an LLM that translates LaTeX into executable SymPy code, scored by actually running the code and checking the numbers — then improved it with RL using code-execution rewards (GRPO), the same trick the Math2Code competition winners and DeepSeek-R1 use."*
- **Show, in order:** (1) the frozen test set + protocol (integrity), (2) the baseline table with CIs (measurement), (3) one before/after GRPO trajectory (the "aha"), (4) the sandbox defense-in-depth diagram (safety), (5) CI/CD that runs the eval nightly (engineering), (6) the Spaces demo (delivery).
- **Interview questions you'll be ready for:** "How do you know the model is actually right?" (execution vs. string match), "Why GRPO?" (verifiable rewards, no RM), "How do you prevent reward hacking?" (§8), "Why not trust LLM-generated data?" (sympy self-verification), "How would you scale this?" (verl, 32B, multi-turn).
- **Honesty rule:** never present the current README claims as done — the upgrade *is* the project narrative: "2024 capstone → 2025 production-grade, with the eval that proves it."

---

## Appendix A — Files referenced

- `configs/train.yaml` — SFT config (r=32, α=64, 5 epochs, bf16, W&B)
- `src/data/generate.py`, `src/data/curate.py` — generation/curation (broken data contract)
- `src/model/train.py` — SFT (KeyError on `python_code`)
- `src/evaluation/eval.py` — non-functional "pass@1"
- `src/serve/api.py`, `src/serve/app.py` — FastAPI + Gradio
- `tests/` — 3 files, mocked E2B/vLLM; ruff/mypy-failing
- `.github/workflows/ci.yml` — lint+typecheck only, both red
- `Dockerfile`, `docker-compose.yml` — broken build, GPU-hardcoded
- `pyproject.toml`, `requirements.txt` — split/duplicated deps; training deps uninstallable
- `data/final/synthetic_data_final.json` — 5,013 rows, no test cases, 30 dup pairs
- `notebooks/` — original capstone incl. the *correct* isclose-based judge (rescue this)

## Appendix B — Verified toolchain versions (Aug 2025)

`trl 1.9.2` · `math-verify 0.9.0` · `verl 0.8.0` · `OpenRLHF 0.10.4` · `vllm 0.26.0` · `e2b-code-interpreter 2.9.0` · `evalplus 0.3.1` · `lm-eval 0.4.12` · `inspect-ai 0.3.253` · `deepeval 4.1.5` · `transformers 5.14.1` · `peft 0.20.0` · `datasets 5.0.1` · `dvc 3.67.1`
