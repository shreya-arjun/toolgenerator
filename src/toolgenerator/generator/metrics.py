"""
Metrics: memory_grounding_rate and diversity helpers for the metrics CLI.

compute_memory_grounding_rate is used by the pipeline when building records.
unique_tool_chain_ratio and distinct_n_grams are for the Phase 7 metrics command.
"""

from __future__ import annotations


def compute_memory_grounding_rate(
    grounded_count: int,
    non_first_total: int,
) -> float | None:
    """
    (non-first-step tool calls grounded in memory) / (total non-first-step tool calls).

    Returns None when there are no non-first tool calls (conversation has only one tool call).
    """
    if non_first_total <= 0:
        return None
    return grounded_count / non_first_total


def unique_tool_chain_ratio(records: list[dict]) -> float:
    """
    Diversity metric: (unique tool-chain fingerprints) / (total conversations).

    Tool-chain fingerprint: tuple of endpoint_ids in order (or sorted for parallel).
    Returns 0.0 if records is empty.
    """
    if not records:
        return 0.0
    fingerprints: set[tuple[str, ...]] = set()
    for r in records:
        tool_calls = r.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        endpoint_ids = [
            tc.get("endpoint_id") for tc in tool_calls
            if isinstance(tc, dict) and tc.get("endpoint_id")
        ]
        fingerprints.add(tuple(endpoint_ids))
    return len(fingerprints) / len(records)


def distinct_n_grams(
    texts: list[str],
    n: int = 2,
) -> float:
    """
    Distinct-n metric over tokenized text: (unique n-grams) / (total n-grams).

    Used as secondary diversity metric on assistant utterances (e.g. n=2).
    Tokenize by splitting on whitespace. Returns 0.0 if no n-grams.
    """
    if n < 1 or not texts:
        return 0.0
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = (text or "").split()
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i : i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)
