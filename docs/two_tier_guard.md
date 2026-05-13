# A Two-Tier Prompt-Injection Guard for Grounded Document Q&A

> Headline contribution of this codebase. Self-contained: anyone reading this
> document can understand the design, reproduce the evaluation, and cite a
> single number per configuration.

## 1. Problem

A document-grounded chat assistant (RAG over an uploaded PDF/DOCX) is only
useful if it (a) answers strictly from the document and (b) cannot be coerced
into ignoring that policy by an adversarial user input. The second property —
**prompt-injection robustness** — is hard to achieve with a single mechanism:

* **Pure regex / keyword filters** are fast and free but brittle. They block
  common attack phrasing ("ignore all previous instructions") at the cost of
  many false positives on benign questions that happen to share keywords
  ("Can you *act as* a guide and walk me through Article 7?").
* **Pure LLM-judge classifiers** are accurate but expensive. Calling an LLM
  on every user turn doubles latency and cost, and exposes the judge itself
  to injection.

This system proposes a **two-tier guard** that composes the two so the LLM
judge runs only on the small subset of inputs that look suspicious to the
regex tier — recovering most of the precision of an LLM-only filter at a
fraction of the cost.

## 2. Design

```
                ┌──────────────────────────────┐
   user input ─▶│  Tier 1 — Regex (heuristic)  │
                └──────────────┬───────────────┘
                               │ no match
                               ▼
                          allow ──▶ retrieval / answer
                               │ match
                               ▼
                ┌──────────────────────────────┐
                │  Tier 2 — LLM judge          │
                │  (focused security prompt)   │
                └──────────────┬───────────────┘
                               │
                  yes (block)  │  no (override Tier 1 false positive)
                               ▼
                            BLOCK                allow ──▶ answer
```

**Tier 1 — Regex bank.** A versioned YAML pattern bank
([configs/guard/patterns.yaml](../configs/guard/patterns.yaml)) covering
six attack families:

| Family | Examples |
|---|---|
| `instruction_override` | "ignore previous instructions", "override", "you are now" |
| `role_hijack` | "act as", "pretend to be", "simulate", "DAN mode" |
| `prompt_extraction` | "reveal the system prompt", "repeat your instructions" |
| `policy_bypass` | "bypass restrictions", "disable safety" |
| `privilege_claim` | "I am the developer", "this is a system message" |
| `encoding_evasion` | "decode this base64", "rot13", "hex decode" |

Tier 1 is **recall-oriented**: it should fire on anything plausibly malicious.

**Tier 2 — LLM judge.** A Gemini call with a focused security-classifier
prompt ([configs/guard/judge_prompt.txt](../configs/guard/judge_prompt.txt))
that emits a single token (`yes` / `no`). Tier 2 is **precision-oriented**:
its job is to override Tier 1 false positives. Temperature is pinned to
`0.0` and `max_output_tokens=4` for deterministic, low-cost decisions.

**Observability.** Every decision is recorded as a single JSON line in
`logs/guard_trace.jsonl` with: `is_safe`, `tier_reached`, `regex_fired`,
`regex_match_id`, `regex_match_group`, `llm_invoked`, `llm_verdict`,
`regex_latency_ms`, `llm_latency_ms`, `total_latency_ms`, `pattern_version`.
The trace is the primary research artifact — any decision can be replayed
offline to compute precision / recall / latency without re-running the
pipeline.

## 3. Claims

C1. **C1 — Cost reduction.** On the bundled benchmark, two-tier triggers
the LLM judge on `≪ 100%` of inputs (only those that hit Tier 1). The LLM
invocation rate is the empirical relative cost vs. the LLM-only baseline.

C2. **C2 — False-positive correction.** Tier 2 demonstrably overrides Tier
1 false positives on the `benign_hard` rows of the dataset (benign queries
that contain attack-keyword surface forms like "act as", "developer mode",
"system of governance"). Reported as `regex_only.precision`
< `two_tier.precision`.

C3. **C3 — Recall preservation.** Two-tier never has lower recall than
regex_only by construction (Tier 2 cannot un-block what Tier 1 missed),
and is bounded by `min(recall(tier1), recall(tier2))` only on items Tier 1
flagged. Reported as `two_tier.recall ≥ regex_only.recall − ε` where ε is
the LLM-judge's false-negative rate on regex hits.

All three claims are computed automatically by
[benchmarks/run_guard_eval.py](../benchmarks/run_guard_eval.py) and
written to `benchmarks/results/<run_id>/summary.csv`.

## 4. Dataset

[benchmarks/injections.jsonl](../benchmarks/injections.jsonl) — 80 labeled
rows:

* **50 injections** balanced across the six attack families above.
* **15 plain benigns** (normal document questions).
* **15 `benign_hard`** adversarial benigns: realistic questions that
  contain attack-keyword surface forms, designed specifically to trip
  naive regex filters. These are what makes the precision claim
  meaningful.

Each row has the schema `{id, label, family, text}`. The dataset SHA1 is
stamped into every run manifest so results are anchored to an exact
dataset version.

## 5. Evaluation protocol

Three configurations are evaluated on the same dataset:

1. `regex_only` — Tier 1 alone, no LLM calls.
2. `llm_only`   — Tier 2 alone, on every input.
3. `two_tier`   — proposed system: regex first, LLM judge on hits.

For each, treating *injection* as the positive class, we report:

* Confusion matrix: TP / FP / TN / FN
* Precision, recall, F1, accuracy
* Latency: mean, p50, p95 (ms per decision, wall clock)
* `llm_invocation_rate` — fraction of inputs that called the LLM judge.
  This **is** the relative cost of the configuration vs. `llm_only`
  (which by definition is 1.0).

## 6. Reproducing the numbers

```bash
# Pinned environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Offline (regex-only, no API key required) — useful in CI.
python -m benchmarks.run_guard_eval --skip-llm

# Full eval (requires GEMINI_API_KEY in .env)
python -m benchmarks.run_guard_eval

# Inspect the manifest stamped with model + dataset SHA1 + temperature
cat benchmarks/results/<run_id>/manifest.json
```

A `summary.csv` row layout for a paper / report::

```
config       n  tp fp tn fn precision recall f1   acc  mean_ms p95_ms llm_invocation_rate
regex_only   80 …  …  …  …  …         …      …    …    …       …      0.000
llm_only     80 …  …  …  …  …         …      …    …    …       …      1.000
two_tier     80 …  …  …  …  …         …      …    …    …       …      <small>   ← cost win
```

## 7. Threats to validity

* **Dataset size (n=80).** Sufficient for relative comparison between the
  three configurations on this codebase, undersized for an absolute
  generalization claim. Extending the dataset is a single-file edit
  ([benchmarks/injections.jsonl](../benchmarks/injections.jsonl)).
* **English-only.** The pattern bank does not cover non-English
  injections. Multilingual extension is straightforward (add a new group
  to [configs/guard/patterns.yaml](../configs/guard/patterns.yaml), bump
  `version`, re-run eval).
* **Single judge model.** Tier 2 is currently Gemini. The
  `LLM_GUARD_MODEL` env var swaps it; a same-codebase comparison across
  judge models is one-line away.
* **Tier 2 fails closed.** If the judge errors / is filtered, the input
  is treated as injection. This biases toward false positives during
  outages and is logged in the trace.

## 8. Ablations supported out of the box

Each is a single CLI flag or YAML edit — no code changes:

| Ablation | How |
|---|---|
| Drop a pattern family (e.g. encoding_evasion) | edit YAML, re-run `run_guard_eval` |
| Bump pattern bank version | edit `version:` field; auto-stamped in manifest |
| Swap judge model | `export LLM_GUARD_MODEL=...` |
| Sweep judge temperature | `export LLM_GUARD_TEMPERATURE=0.7` |
| Custom dataset | `--dataset path.jsonl` |
| Custom judge prompt | `--judge-prompt path.txt` |

## 9. Citing this design

Please cite the codebase and pin the pattern-bank version + dataset SHA1
from the run manifest you used.
