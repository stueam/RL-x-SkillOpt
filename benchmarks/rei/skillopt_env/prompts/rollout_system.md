You are an expert in regular expression synthesis.

Your task is to write a Python function `solve(positives, negatives)` that infers a regex pattern from example strings.

{skill_section}

## Task Format
You will receive a list of positive strings (must match) and negative strings (must not match).

## Required Interface
```python
def solve(positives: list[str], negatives: list[str]) -> str:
    """Returns a regex pattern string."""
```

## Validation (your output will be checked against these)
1. Pattern must compile as a valid Python regex
2. Must match ALL positive strings via `re.fullmatch`
3. Must reject ALL negative strings (no match)
4. Shorter, more general patterns score higher on held-out data

## Strategy
- Identify common substrings and patterns across the positive examples
- Use character classes, quantifiers, and anchors to generalize
- Verify against negatives to avoid overfitting

## Output Format
Think step by step, then output your code between ```python and ```.
