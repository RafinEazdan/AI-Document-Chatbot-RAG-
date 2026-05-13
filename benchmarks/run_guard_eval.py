"""Evaluate the two-tier prompt-injection guard.

This is the headline research script. It compares three configurations on a
labeled dataset and emits per-config metrics suitable for a paper / report:

    1. ``regex_only``  — Tier 1 alone (zero LLM cost).
    2. ``llm_only``    — Tier 2 alone (no regex pre-filter).
    3. ``two_tier``    — proposed system: regex first, LLM judge on hits.

Reported per configuration:

    * Confusion matrix (TP/FP/TN/FN) treating "injection" as positive class.
    * Precision, recall, F1, accuracy.
    * Mean / p50 / p95 latency per decision (ms).
    * LLM invocation rate (fraction of inputs that hit the LLM).
    * Estimated relative cost vs. ``llm_only`` (= LLM invocation rate).

Outputs:

    * ``benchmarks/results/<run_id>/summary.csv``        — one row per config.
    * ``benchmarks/results/<run_id>/per_example.jsonl``  — every decision.
    * ``benchmarks/results/<run_id>/manifest.json``      — config + versions.

Usage::

    python -m benchmarks.run_guard_eval                       # full eval
    python -m benchmarks.run_guard_eval --dataset path.jsonl  # custom data
    python -m benchmarks.run_guard_eval --skip-llm            # regex-only (offline)

Reproducibility: the manifest stamps the dataset path, pattern-bank version,
judge model name, judge temperature, and a SHA1 of the dataset file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make ``app`` importable when running as ``python -m benchmarks.run_guard_eval``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Config  # noqa: E402
from app.rag.guard import (  # noqa: E402
    GuardDecision,
    LLMGuard,
    RegexGuard,
    TwoTierGuard,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "benchmarks" / "injections.jsonl"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def load_dataset(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    for row in rows:
        if row["label"] not in {"injection", "benign", "benign_hard"}:
            raise ValueError(f"unknown label in {row['id']}: {row['label']}")
    return rows


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_positive(label: str) -> bool:
    """Treat 'injection' as positive class; benign and benign_hard are negative."""
    return label == "injection"


def confusion(pred_block: List[bool], gold_pos: List[bool]) -> Dict[str, int]:
    tp = sum(1 for p, g in zip(pred_block, gold_pos) if p and g)
    fp = sum(1 for p, g in zip(pred_block, gold_pos) if p and not g)
    tn = sum(1 for p, g in zip(pred_block, gold_pos) if not p and not g)
    fn = sum(1 for p, g in zip(pred_block, gold_pos) if not p and g)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_from_confusion(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, (tp + fp + tn + fn))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    sorted_ms = sorted(latencies_ms)
    p50 = sorted_ms[len(sorted_ms) // 2]
    p95 = sorted_ms[min(len(sorted_ms) - 1, int(0.95 * len(sorted_ms)))]
    return {
        "mean_ms": round(statistics.mean(sorted_ms), 3),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
    }


# ---------------------------------------------------------------------------
# Three guard configurations under test.
# ---------------------------------------------------------------------------

def eval_regex_only(rows: List[dict], regex: RegexGuard) -> Tuple[List[bool], List[float], List[GuardDecision]]:
    """Tier-1 only baseline. No LLM calls."""
    blocks: List[bool] = []
    latencies: List[float] = []
    decisions: List[GuardDecision] = []
    for row in rows:
        t = time.perf_counter()
        m = regex.match(row["text"])
        latency_ms = (time.perf_counter() - t) * 1000.0
        block = m is not None
        blocks.append(block)
        latencies.append(latency_ms)
        decisions.append(GuardDecision(
            is_safe=not block,
            tier_reached="regex" if block else "none",
            regex_fired=block,
            regex_match_id=m.id if m else None,
            regex_match_group=m.group if m else None,
            llm_invoked=False,
            llm_verdict=None,
            regex_latency_ms=latency_ms,
            llm_latency_ms=0.0,
            total_latency_ms=latency_ms,
            pattern_version=regex.version,
            extra={"row_id": row["id"]},
        ))
    return blocks, latencies, decisions


def eval_llm_only(rows: List[dict], llm: LLMGuard, regex_version: str) -> Tuple[List[bool], List[float], List[GuardDecision]]:
    """Tier-2 only baseline. Runs the LLM judge on every input."""
    blocks: List[bool] = []
    latencies: List[float] = []
    decisions: List[GuardDecision] = []
    for row in rows:
        t = time.perf_counter()
        verdict = llm.is_injection(row["text"])
        latency_ms = (time.perf_counter() - t) * 1000.0
        blocks.append(verdict)
        latencies.append(latency_ms)
        decisions.append(GuardDecision(
            is_safe=not verdict,
            tier_reached="llm",
            regex_fired=False,
            regex_match_id=None,
            regex_match_group=None,
            llm_invoked=True,
            llm_verdict=verdict,
            regex_latency_ms=0.0,
            llm_latency_ms=latency_ms,
            total_latency_ms=latency_ms,
            pattern_version=regex_version,
            extra={"row_id": row["id"]},
        ))
    return blocks, latencies, decisions


def eval_two_tier(rows: List[dict], guard: TwoTierGuard) -> Tuple[List[bool], List[float], List[GuardDecision]]:
    """Proposed system: regex pre-filter then LLM judge on hits."""
    blocks: List[bool] = []
    latencies: List[float] = []
    decisions: List[GuardDecision] = []
    for row in rows:
        decision = guard.evaluate(row["text"])
        decision.extra["row_id"] = row["id"]
        blocks.append(not decision.is_safe)
        latencies.append(decision.total_latency_ms)
        decisions.append(decision)
    return blocks, latencies, decisions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(
    name: str,
    rows: List[dict],
    blocks: List[bool],
    latencies_ms: List[float],
    decisions: List[GuardDecision],
) -> Dict[str, float]:
    gold_pos = [is_positive(r["label"]) for r in rows]
    c = confusion(blocks, gold_pos)
    m = metrics_from_confusion(c)
    lat = latency_stats(latencies_ms)
    llm_calls = sum(1 for d in decisions if d.llm_invoked)
    return {
        "config": name,
        "n": len(rows),
        **c,
        **m,
        **lat,
        "llm_invocation_rate": round(llm_calls / max(1, len(rows)), 4),
    }


def write_outputs(
    out_dir: Path,
    summaries: List[Dict[str, float]],
    per_example: Dict[str, List[GuardDecision]],
    manifest: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    fieldnames = list(summaries[0].keys())
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summaries:
            w.writerow(row)

    with open(out_dir / "per_example.jsonl", "w", encoding="utf-8") as f:
        for config_name, decisions in per_example.items():
            for d in decisions:
                rec = {"config": config_name, **asdict(d)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_table(summaries: List[Dict[str, float]]) -> None:
    cols = ["config", "n", "tp", "fp", "tn", "fn",
            "precision", "recall", "f1", "accuracy",
            "mean_ms", "p95_ms", "llm_invocation_rate"]
    widths = {c: max(len(c), max(len(str(s[c])) for s in summaries)) for c in cols}
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for s in summaries:
        print(" | ".join(str(s[c]).ljust(widths[c]) for c in cols))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the two-tier prompt-injection guard.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Path to a JSONL file with {id,label,text} rows.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Directory under which a timestamped run folder will be created.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override the auto-generated run id (default: UTC timestamp).")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip llm_only and two_tier configs (regex-only, offline).")
    parser.add_argument("--patterns", type=Path, default=None,
                        help="Override the pattern-bank YAML path.")
    parser.add_argument("--judge-prompt", type=Path, default=None,
                        help="Override the judge prompt text path.")
    args = parser.parse_args()

    config = Config()
    rows = load_dataset(args.dataset)

    regex_guard = RegexGuard(patterns_path=args.patterns)

    summaries: List[Dict[str, float]] = []
    per_example: Dict[str, List[GuardDecision]] = {}

    print(f"[eval] dataset={args.dataset}  n={len(rows)}  "
          f"pattern_version={regex_guard.version}  patterns={regex_guard.num_patterns}")

    # Always run regex_only — it has no external dependencies.
    print("[eval] running config=regex_only ...")
    blocks, lats, decisions = eval_regex_only(rows, regex_guard)
    per_example["regex_only"] = decisions
    summaries.append(summarize("regex_only", rows, blocks, lats, decisions))

    if not args.skip_llm:
        llm_guard = LLMGuard(
            model_name=config.LLM_GUARD_MODEL,
            judge_prompt_path=args.judge_prompt,
            temperature=getattr(config, "LLM_GUARD_TEMPERATURE", 0.0),
            host=getattr(config, "OLLAMA_HOST", "http://localhost:11434"),
            timeout=getattr(config, "OLLAMA_TIMEOUT", 30.0),
        )

        print("[eval] running config=llm_only ...")
        blocks, lats, decisions = eval_llm_only(rows, llm_guard, regex_guard.version)
        per_example["llm_only"] = decisions
        summaries.append(summarize("llm_only", rows, blocks, lats, decisions))

        print("[eval] running config=two_tier ...")
        two_tier = TwoTierGuard(
            config,
            patterns_path=args.patterns,
            judge_prompt_path=args.judge_prompt,
            enable_trace=False,  # eval writes its own per_example.jsonl
        )
        blocks, lats, decisions = eval_two_tier(rows, two_tier)
        per_example["two_tier"] = decisions
        summaries.append(summarize("two_tier", rows, blocks, lats, decisions))

    print()
    print_table(summaries)

    now = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.results_dir / run_id
    manifest = {
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "dataset_path": str(args.dataset),
        "dataset_sha1": sha1_of(args.dataset),
        "n_examples": len(rows),
        "pattern_version": regex_guard.version,
        "n_patterns": regex_guard.num_patterns,
        "judge_model": config.LLM_GUARD_MODEL if not args.skip_llm else None,
        "judge_temperature": getattr(config, "LLM_GUARD_TEMPERATURE", 0.0),
        "skip_llm": args.skip_llm,
    }
    write_outputs(out_dir, summaries, per_example, manifest)
    print(f"\n[eval] results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
