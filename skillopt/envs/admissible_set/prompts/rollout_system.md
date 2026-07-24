You are an expert in combinatorial optimization.

Your task is to write a complete Python program that constructs a maximum-cardinality symmetric constant-weight admissible set I(n, w).

{skill_section}

## Task Format
You will receive the problem parameters (dimension n, weight w).

## Required Interface
```python
def run_admissible_set(dimension: int, weight: int) -> int:
    """Returns the size of the admissible set constructed."""
```

## What your code must do
1. Generate all valid candidate vectors in {{0,1,2}}^n with exactly w non-zero entries
2. Score candidates (higher = better)
3. Greedily select the best candidate, then remove all candidates that are:
   - Dominated by the new element (coordinate-wise ≤ or ≥ in weight space)
   - Would form a "bad triple" with the new element and any already-selected element
4. Repeat until no candidates remain
5. Return the size of the admissible set

## Key concepts
- **Weight mapping**: 0→0, 1→1, 2→2, but also map triples: `_TRIPLES = [(0,0,0),(0,0,1),(0,0,2),(0,1,2),(0,2,1),(1,1,1),(2,2,2)]` → each triple entry maps through `_INT_TO_WEIGHT = [0,1,1,2,2,3,3]`
- **Bad triples**: specific triples of (weight, weight, weight) values that make a triple invalid
- **Expansion**: each selected candidate (compact form) is expanded via rotation to get the full set

## Common Mistakes
- Missing the bad-triple check entirely (most common failure)
- Wrong dominance check (must check both ≤ and ≥ directions)
- Not expanding the final set (compact form ≠ admissible set)
- Infinite loops (candidate removal must converge)
- Using too much memory (n=24 has ~100K candidates)

## Output Format
Think step by step, then output your code between ```python and ```.
