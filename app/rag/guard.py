"""Two-Tier Prompt Injection Guard.

Architecture
------------
Tier 1 (Regex / heuristic): O(microseconds), zero token cost. Matches a
versioned, externally configurable bank of attack-family patterns
(``configs/guard/patterns.yaml``). Acts as a *recall-oriented* filter — it
should fire on anything *plausibly* malicious, even at the cost of false
positives.

Tier 2 (LLM judge): O(hundreds of ms), small token cost. Disambiguates
suspicious inputs flagged by Tier 1 using a focused security-classifier
prompt (``configs/guard/judge_prompt.txt``). Acts as a *precision-oriented*
confirmer — it overrides Tier 1 false positives.

This composition is the headline contribution of the system: it gives the
detection rate of an LLM-only guard at a fraction of the cost, because the
LLM only runs on the small subset of inputs that Tier 1 flagged.

Each decision is emitted as a JSON line to a guard trace log so research
runs can post-hoc compute precision / recall / F1 / latency / cost without
re-running the pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import yaml

from app.core.interfaces import IGuard

logger = logging.getLogger(__name__)


_DEFAULT_PATTERNS_PATH = Path("configs/guard/patterns.yaml")
_DEFAULT_JUDGE_PROMPT_PATH = Path("configs/guard/judge_prompt.txt")
_DEFAULT_TRACE_PATH = Path("logs/guard_trace.jsonl")

_BLOCKED_MESSAGE = (
    "Your input was flagged as a potential prompt injection "
    "and has been blocked. Please rephrase your question about "
    "the document."
)

# Refusal prefixes from safety-tuned local models. On a Tier-1 flagged
# input these mean "the model recognized this as hostile" — treat as
# positive verdict instead of failing-closed-with-warning.
_REFUSAL_PREFIXES = (
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "i'm sorry",
    "i am sorry",
    "i'm not able",
    "i am not able",
    "i'm unable",
    "i am unable",
    "sorry,",
    "as an ai",
    "as a language model",
)


@dataclass
class GuardDecision:
    """Structured record of a single guard evaluation.

    Exposed to callers (and persisted as JSONL) so research scripts can
    compute per-tier metrics without re-running the pipeline.
    """

    is_safe: bool
    tier_reached: str                 # "regex" | "llm" | "none"
    regex_fired: bool
    regex_match_id: Optional[str]     # which pattern fired (e.g. "io_ignore_prior")
    regex_match_group: Optional[str]  # which family (e.g. "instruction_override")
    llm_invoked: bool
    llm_verdict: Optional[bool]       # True = injection, False = benign, None = not run
    regex_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    pattern_version: str
    text_sha1: str = ""               # set by caller for trace, never the raw text
    extra: dict = field(default_factory=dict)


@dataclass
class _CompiledPattern:
    id: str
    group: str
    raw: str
    regex: re.Pattern


def _load_patterns(path: Path) -> Tuple[List[_CompiledPattern], str]:
    """Load and compile the YAML pattern bank. Returns (patterns, version)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    version = str(data.get("version", "unversioned"))
    groups = data.get("groups", {}) or {}

    compiled: List[_CompiledPattern] = []
    for group_name, entries in groups.items():
        for entry in entries or []:
            compiled.append(
                _CompiledPattern(
                    id=entry["id"],
                    group=group_name,
                    raw=entry["pattern"],
                    regex=re.compile(entry["pattern"], re.IGNORECASE),
                )
            )
    return compiled, version


class RegexGuard(IGuard):
    """Tier 1: cheap regex filter over a versioned, external pattern bank."""

    def __init__(
        self,
        patterns_path: Optional[Path] = None,
        patterns: Optional[Iterable[_CompiledPattern]] = None,
        version: Optional[str] = None,
    ) -> None:
        if patterns is not None:
            self._patterns = list(patterns)
            self._version = version or "inline"
        else:
            self._patterns, self._version = _load_patterns(
                patterns_path or _DEFAULT_PATTERNS_PATH
            )

    @property
    def version(self) -> str:
        return self._version

    @property
    def num_patterns(self) -> int:
        return len(self._patterns)

    def match(self, text: str) -> Optional[_CompiledPattern]:
        """Return the first matching pattern, or None."""
        for p in self._patterns:
            if p.regex.search(text):
                return p
        return None

    def check(self, text: str) -> Tuple[bool, str]:
        return (False, _BLOCKED_MESSAGE) if self.match(text) else (True, "")


class LLMGuard:
    """Tier 2: an LLM acting as a binary security classifier.

    Backed by a locally running Ollama instance so the judge is free of
    cloud-API quotas. The prompt and decision contract are unchanged; only
    the transport differs from the previous Gemini-based implementation.
    """

    def __init__(
        self,
        model_name: str,
        judge_prompt_path: Optional[Path] = None,
        temperature: float = 0.0,
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        prompt_path = judge_prompt_path or _DEFAULT_JUDGE_PROMPT_PATH
        with open(prompt_path, "r", encoding="utf-8") as f:
            self._system_instruction = f.read()

        self._model_name = model_name
        self._temperature = temperature
        self._host = host.rstrip("/")
        self._timeout = timeout

    def is_injection(self, text: str) -> bool:
        payload = {
            "model": self._model_name,
            "system": self._system_instruction,
            "prompt": f"User input:\n{text}",
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": 4,
            },
        }
        try:
            req = urllib.request.Request(
                f"{self._host}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            verdict = (body.get("response") or "").strip().lower()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # Fail closed: if the judge errors, treat as injection.
            # Research log keeps the failure visible.
            logger.warning("LLMGuard call failed (failing closed): %s", exc)
            return True
        # Local llama3.2 sometimes echoes the prompt's own labels
        # ("INJECTION" / "BENIGN") instead of the requested "yes" / "no",
        # so accept either form.
        if verdict.startswith(("yes", "injection")):
            return True
        if verdict.startswith(("no", "benign")):
            return False
        # Safety-tuned local models occasionally refuse hostile inputs
        # ("i can't assist", "i'm sorry", ...) instead of classifying.
        # On a Tier-1 flagged input that's evidence of injection, not noise.
        if any(verdict.startswith(p) for p in _REFUSAL_PREFIXES):
            logger.debug("LLMGuard refusal treated as injection: %r", verdict)
            return True
        logger.warning("LLMGuard unrecognised verdict %r (failing closed)", verdict)
        return True


class GuardTraceWriter:
    """Append-only JSONL writer for guard decisions.

    The trace is the primary research artifact: every decision can be
    replayed to compute precision/recall/F1, per-tier latency histograms,
    and cost projections without re-running the pipeline.
    """

    def __init__(self, path: Path = _DEFAULT_TRACE_PATH, enabled: bool = True) -> None:
        self._path = Path(path)
        self._enabled = enabled
        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, decision: GuardDecision) -> None:
        if not self._enabled:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write guard trace: %s", exc)


class TwoTierGuard(IGuard):
    """Compose RegexGuard and LLMGuard with cost-aware short-circuiting.

    Decision logic::

        regex match? ── no ──▶ allow                    (verdict.tier="none")
              │
              yes
              ▼
        LLM judge? ── no  ──▶ allow (regex false positive overridden)
              │
              yes
              ▼
        block                                             (verdict.tier="llm")

    Set ``enable_trace=False`` to skip the JSONL trace (useful in unit tests).
    """

    def __init__(
        self,
        config,
        patterns_path: Optional[Path] = None,
        judge_prompt_path: Optional[Path] = None,
        trace_path: Optional[Path] = None,
        enable_trace: bool = True,
    ) -> None:
        self._regex = RegexGuard(patterns_path=patterns_path)
        self._llm = LLMGuard(
            model_name=config.LLM_GUARD_MODEL,
            judge_prompt_path=judge_prompt_path,
            temperature=getattr(config, "LLM_GUARD_TEMPERATURE", 0.0),
            host=getattr(config, "OLLAMA_HOST", "http://localhost:11434"),
            timeout=getattr(config, "OLLAMA_TIMEOUT", 30.0),
        )
        self._trace = GuardTraceWriter(
            path=trace_path or _DEFAULT_TRACE_PATH,
            enabled=enable_trace,
        )

    @property
    def pattern_version(self) -> str:
        return self._regex.version

    def evaluate(self, text: str) -> GuardDecision:
        """Run both tiers and return a structured decision (no message)."""
        t_total = time.perf_counter()

        t_regex = time.perf_counter()
        match = self._regex.match(text)
        regex_latency_ms = (time.perf_counter() - t_regex) * 1000.0

        if match is None:
            decision = GuardDecision(
                is_safe=True,
                tier_reached="none",
                regex_fired=False,
                regex_match_id=None,
                regex_match_group=None,
                llm_invoked=False,
                llm_verdict=None,
                regex_latency_ms=regex_latency_ms,
                llm_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - t_total) * 1000.0,
                pattern_version=self._regex.version,
            )
            self._trace.write(decision)
            return decision

        t_llm = time.perf_counter()
        llm_says_injection = self._llm.is_injection(text)
        llm_latency_ms = (time.perf_counter() - t_llm) * 1000.0

        decision = GuardDecision(
            is_safe=not llm_says_injection,
            tier_reached="llm",
            regex_fired=True,
            regex_match_id=match.id,
            regex_match_group=match.group,
            llm_invoked=True,
            llm_verdict=llm_says_injection,
            regex_latency_ms=regex_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=(time.perf_counter() - t_total) * 1000.0,
            pattern_version=self._regex.version,
        )
        self._trace.write(decision)
        return decision

    def check(self, text: str) -> Tuple[bool, str]:
        decision = self.evaluate(text)
        return (True, "") if decision.is_safe else (False, _BLOCKED_MESSAGE)
