"""Simple A/B testing utilities for SIGLA.

This module provides a lightweight way to compare *baseline* answers with
SIGLA‐powered (context-merged) answers on a small evaluation set.

The evaluation uses optional metrics (ROUGE-L, BLEU) when corresponding
packages are installed; otherwise it falls back to a trivial token‐overlap
score so that the code can run without heavyweight NLP dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from .core import CapsuleStore, merge_capsules, MissingDependencyError

# Optional metrics
try:
    from rouge_score import rouge_scorer  # type: ignore
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_overlap(a: str, b: str) -> float:
    """Return ratio of shared tokens between *a* and *b* (Jaccard)."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _compute_metrics(pred: str, ref: str) -> Dict[str, float]:
    """Compute metric dict for single prediction/reference pair."""
    metrics: Dict[str, float] = {
        "overlap": _simple_overlap(pred, ref)
    }
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        metrics["rougeL"] = score
    if BLEU_AVAILABLE:
        smoothie = SmoothingFunction().method4  # type: ignore[attr-defined]
        metrics["bleu"] = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)  # type: ignore[arg-type]
    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ab_test(
    store: CapsuleStore,
    dataset: List[Dict[str, str]],
    *,
    top_k: int = 5,
    temperature: float = 1.0,
    baseline: str = "echo",
) -> Dict[str, Any]:
    """Run A/B test and return aggregated metrics.

    Parameters
    ----------
    store
        CapsuleStore with knowledge base.
    dataset
        List of dicts with keys ``question`` and ``answer``.
    top_k, temperature
        Retrieval / merge parameters for SIGLA pipeline.
    baseline
        "echo" (return question) or "none" (empty string).
    """
    results: List[Tuple[Dict[str, float], Dict[str, float]]] = []

    for item in dataset:
        q = item["question"]
        ref = item.get("answer", "")

        # SIGLA pipeline
        caps = store.query(q, top_k=top_k)
        sigla_answer = merge_capsules(caps, temperature=temperature)

        # Baseline
        if baseline == "echo":
            base_answer = q
        else:
            base_answer = ""

        sigla_metrics = _compute_metrics(sigla_answer, ref)
        base_metrics = _compute_metrics(base_answer, ref)
        results.append((sigla_metrics, base_metrics))

    # Aggregate
    agg: Dict[str, Dict[str, float]] = {"sigla": {}, "baseline": {}}
    for sigm, basm in results:
        for k, v in sigm.items():
            agg["sigla"][k] = agg["sigla"].get(k, 0.0) + v
        for k, v in basm.items():
            agg["baseline"][k] = agg["baseline"].get(k, 0.0) + v

    n = len(results) or 1
    for side in agg.values():
        for k in side:
            side[k] /= n
    return agg


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def cli(args) -> None:  # pragma: no cover
    """Entry point for sigla abtest CLI (wired in scripts.py)."""
    if not Path(args.dataset).exists():
        print(f"dataset not found: {args.dataset}")
        return
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("dataset must be list of objects with 'question' and 'answer'")
        return

    try:
        store = CapsuleStore()
        store.load(args.store)
    except MissingDependencyError as e:
        print(f"error: {e}")
        return

    summary = run_ab_test(store, data, top_k=args.top_k, temperature=args.temperature, baseline=args.baseline)

    print("A/B test summary (avg):")
    for side in ("sigla", "baseline"):
        print(f"  {side}:")
        for k, v in sorted(summary[side].items()):
            print(f"    {k:8}: {v:.3f}") 
