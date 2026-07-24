# Cap Set Construction

Construct a cap set in F_3^n: a subset A of {0,1,2}^n such that no three distinct elements form an arithmetic progression.

Formally: for all distinct x, y, z in A, it is NOT the case that x + y + z ≡ 0 (mod 3) componentwise.

## REQUIRED interface

```python
def construct(n: int, seed: int) -> np.ndarray:
    """Returns a 2D numpy array of shape (m, n), dtype int, entries in {0, 1, 2}.
    m = |A|. Maximize m.
    """
```

## Reliable utilities — copy these into your code exactly

### Check if a set is a valid cap set

```python
def is_cap_set(A):
    """Returns True if A (m, n) is a valid cap set."""
    aset = set(map(tuple, A))
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            z = tuple(np.mod(-(A[i] + A[j]), 3))
            if z in aset and z != tuple(A[i]) and z != tuple(A[j]):
                return False
    return True
```

### Generate all candidates

```python
def all_candidates(n):
    """Return np.ndarray of shape (3**n, n) with all vectors in F_3^n."""
    return np.array(list(product([0, 1, 2], repeat=n)), dtype=int)
```

### Greedy construction from ordered candidates

```python
def greedy_build(candidates):
    """Greedily build a cap set from candidates (in given order).
    Returns np.ndarray shape (m, n)."""
    selected = []
    aset = set()
    for v in candidates:
        ok = True
        for a in selected:
            z = tuple(np.mod(-(v + a), 3))
            if z in aset and z != tuple(v) and z != tuple(a):
                ok = False
                break
        if ok:
            selected.append(v)
            aset.add(tuple(v))
    return np.array(selected)
```

## Strategy — you fill in `construct(n, seed)`

Use this structure:

```python
def construct(n, seed):
    rng = np.random.RandomState(seed)

    # Generate all candidates
    cand = all_candidates(n)

    # === YOUR STRATEGY HERE ===
    # Try different candidate orderings. Examples:
    # 1. Random shuffle: rng.shuffle(cand) → greedy_build(cand)
    # 2. Multiple shuffles: try several seeds, return the largest
    # 3. Score-sort: assign scores, sort descending, then greedy
    # 4. Reverse: start from last candidate, go backwards
    # 5. Structured: interleave different "types" of vectors

    best = greedy_build(cand)
    best_size = len(best)

    # Try multiple shuffles for better results
    for _ in range(20):
        rng.shuffle(cand)
        result = greedy_build(cand)
        if len(result) > best_size:
            best = result
            best_size = len(result)

    # Fallback: if nothing works, return the best found
    assert best.shape[1] == n, f"Wrong shape: {best.shape}"
    assert is_cap_set(best), "Invalid cap set produced!"
    return best
```

### CRITICAL DO-NOT-DO (top failures from real runs)

1. **ALWAYS call `is_cap_set()` before returning.** One-off bugs in the selection loop silently produce invalid sets. Validate your output.
2. **MUST use the provided `seed` to produce different results across calls.** Ignoring seed = deterministic output = no diversity. Use `np.random.RandomState(seed)`.
3. **Return a 2D array with shape (m, n).** NOT (n, m), NOT a list of lists.
4. **No duplicate rows.** Greedy construction naturally avoids this, but be careful with manual construction.
5. **All entries must be 0, 1, or 2.** No floats, no -1.
6. **Do NOT hardcode for a specific n.** `construct(n, seed)` must work for any n ≥ 1.
7. **Do NOT redefine `is_cap_set`, `all_candidates`, or `greedy_build`.** Use the provided implementations.

### Additional rules

- Use `np.random.RandomState(seed)` for reproducibility.
- Multiple shuffled trials (20-50) consistently outperform single-pass greedy. This is the single most important optimization.
- For n=4, shuffled greedy typically finds 16-18 (optimal 20). For n=5, 32-38 (optimal 45). For n=6, 64-80 (optimal 112).
- The gap to optimal grows with n, leaving room for more sophisticated strategies (structured orderings, product constructions, iterative refinement).
