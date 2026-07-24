# Regular Expression Inference

Synthesize a regex pattern from positive and negative string examples.

## Interface

```python
def solve(positives: list[str], negatives: list[str]) -> str:
```

- Input: lists of positive (must match) and negative (must reject) strings.
- Output: a regex pattern string.

## Output Requirements

1. Pattern must compile: `re.compile(pattern)` succeeds.
2. Match ALL positives: `re.fullmatch(pattern, p)` for every positive p.
3. Reject ALL negatives: `re.fullmatch(pattern, n)` returns None for every negative n.

Violating any requirement yields Gate = 0.
