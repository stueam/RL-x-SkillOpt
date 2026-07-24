"""Shared REI domain logic: regex validation, held-out scoring."""
import re


def validate_regex(pattern: str, positives: list[str], negatives: list[str]) -> dict:
    """Gate check: valid regex + match all positives + reject all negatives.
    Returns {"gate": 0/1, "error": ""}.
    """
    try:
        re.compile(pattern)
    except re.error as e:
        return {"gate": 0, "error": f"invalid_regex: {e}"}

    for p in positives:
        if not re.fullmatch(pattern, p):
            return {"gate": 0, "error": f"missed_positive: {p!r}"}

    for n in negatives:
        if re.fullmatch(pattern, n):
            return {"gate": 0, "error": f"accepted_negative: {n!r}"}

    return {"gate": 1, "error": ""}


def score_generalization(pattern: str, heldout_positives: list[str], heldout_negatives: list[str]) -> float:
    """Score on held-out data (0-1)."""
    hits = 0
    total = 0
    for p in heldout_positives:
        if re.fullmatch(pattern, p):
            hits += 1
        total += 1
    for n in heldout_negatives:
        if not re.fullmatch(pattern, n):
            hits += 1
        total += 1
    return hits / total if total > 0 else 1.0
