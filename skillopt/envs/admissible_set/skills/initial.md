# Admissible Set

Write a complete `run_admissible_set(dimension, weight) -> int` that constructs a maximum-cardinality symmetric constant-weight admissible set.

## Correctness requirements (hard gate)

Your code MUST do all of the following correctly, or it will return 0:

1. **Generate candidates**: Use compact encoding. Each candidate is a list of length `n/3` with values 0-6 (7 triple types). Only include candidates where `sum(INT_TO_WEIGHT[x] for x in candidate) == weight`.
2. **Greedy selection**: At each step, pick the highest-scored candidate, add it to the set, then remove all candidates that are dominated by it or form bad triples with it and an existing element.
3. **Bad triple check**: For EVERY existing element `ext` and remaining candidate `child`, check if `(ext[g], chosen[g], child[g])` for each group `g` forms a bad triple (all three values sorted must NOT be in the bad triples set).
4. **Dual dominance**: Remove candidate if ALL groups are ≤ chosen OR ALL groups are ≥ chosen (in weight space).
5. **Expansion**: Convert compact representation to full vectors via rotation and Cartesian product.

## Working Python template

```python
import itertools
import numpy as np

TRIPLES = [(0,0,0),(0,0,1),(0,0,2),(0,1,2),(0,2,1),(1,1,1),(2,2,2)]
INT_TO_W = [0,1,1,2,2,3,3]
# There are 36 bad triples - see the EoH reference for the full set

def run_admissible_set(n, w):
    num_groups = n // 3
    # 1. Generate candidates
    candidates = []
    for child in itertools.product(range(7), repeat=num_groups):
        if sum(INT_TO_W[x] for x in child) == w:
            candidates.append(list(child))

    # 2. Score - simple baseline
    expanded = [_expand(c, num_groups) for c in candidates]
    scores = np.array([priority(el, n, w) for el in expanded], dtype=float)

    selected = []
    while candidates:
        best = int(np.argmax(scores))
        chosen = candidates.pop(best)
        scores = np.delete(scores, best)
        selected.append(chosen)
        # Filter remaining candidates
        survivors = []
        new_scores = []
        for i, child in enumerate(candidates):
            # Dominance check
            dominated = all(INT_TO_W[a] <= INT_TO_W[b] for a, b in zip(chosen, child))
            dominates = all(INT_TO_W[a] >= INT_TO_W[b] for a, b in zip(chosen, child))
            if dominated or dominates:
                continue
            # Bad triple check
            bad = False
            for ext in selected[:-1]:  # all previously selected
                if all(_is_bad_triple(ext[g], chosen[g], child[g]) for g in range(num_groups)):
                    bad = True
                    break
            if not bad:
                survivors.append(child)
                new_scores.append(scores[i])
        candidates = survivors
        scores = np.array(new_scores, dtype=float)

    # Expansion
    full = _expand_set(selected, num_groups)
    return len(full)
```

## Scoring strategies

The baseline `sum(abs(x) for x in el) / n` works but is weak. Better strategies:

- **Frequency-aware**: Count how many other candidates would be eliminated if you pick a given element. Pick the element that eliminates the FEWEST future candidates (maximizes remaining search space).
- **Diversity**: Favor elements with diverse triple patterns (more rotation variety).
- **Weight spread**: Favor weight spread evenly across groups rather than concentrated.
- **Combine multiple criteria**: `score = diversity_bonus - elimination_penalty * 0.1`.

## CRITICAL DO-NOT-DO
1. Missing bad triple check → invalid set (score 0.001)
2. Wrong dominance (only checking one direction) → incorrect filtering
3. Forgetting to expand the compact set → wrong reported size
4. Using `for ... in candidates` while modifying `candidates` → infinite loop
5. Memory explosion for n=12 (2401 candidates → fine, but n=24 has 576K)
