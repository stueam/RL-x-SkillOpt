# Circle Packing

Pack 26 circles in the unit square [0,1]x[0,1]. Maximize sum(radii).
No overlaps, all circles strictly inside square, radii non-negative.

## REQUIRED interface

```python
def run_packing():
    """Returns (centers, radii, sum_radii)."""
    # centers: (26, 2) float ndarray in [0,1]
    # radii: (26,) nonnegative float ndarray
    return centers, radii, sum_radii
```

## Reliable utilities — copy these into your code exactly

### Validate a packing

```python
def validate(centers, radii, n):
    if radii.shape != (n,):
        return False, 0.0
    if np.any(radii < 0):
        return False, 0.0
    # Keep centers inside bounds
    if np.any(centers < 0) or np.any(centers > 1):
        return False, 0.0
    # Boundary constraints
    for i in range(n):
        if centers[i,0] - radii[i] < -1e-9:
            return False, 0.0
        if centers[i,1] - radii[i] < -1e-9:
            return False, 0.0
        if centers[i,0] + radii[i] > 1.0 + 1e-9:
            return False, 0.0
        if centers[i,1] + radii[i] > 1.0 + 1e-9:
            return False, 0.0
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return False, 0.0
    return True, float(np.sum(radii))
```

### Repair overlaps — call this after any strategy

```python
def repair(centers, radii, n, max_iter=200):
    radii = np.maximum(radii, 0.001)
    for _ in range(max_iter):
        fixed = True
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    overlap = (radii[i] + radii[j] - dist) / 2.0
                    if overlap > 0:
                        radii[i] = max(0.001, radii[i] - overlap * 0.5)
                        radii[j] = max(0.001, radii[j] - overlap * 0.5)
                        fixed = False
        if fixed:
            break
    # Re-clip to bounds
    for i in range(n):
        max_r = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        radii[i] = min(radii[i], max_r)
    return centers, radii
```

## Strategy — you fill in this function

Your job is to write `run_packing()`. The validation and repair functions above are
reliable — include them and call them, don't reimplement them.

Use this structure:

```python
def generate_candidates(n, seed):
    """Try different initial layouts, return list of (centers, radii)."""
    best = None
    best_sum = 0.0
    rng = np.random.RandomState(seed)

    # === YOUR STRATEGY HERE ===
    # Try different layout strategies. Examples:
    # - Hexagonal grid with jittered centers and equal radii
    # - Greedy: place circles one by one, each at the best free spot
    # - Corner-first: largest circles at corners, fill remaining space
    # - More circles at edges, fewer in center (boundary-locked packing)

    for trial in range(30):
        # 1. Generate an initial (centers, radii) for this trial
        # 2. Repair overlaps: centers, radii = repair(centers, radii, n)
        # 3. Grow radii where there is room
        # 4. Validate: ok, score = validate(centers, radii, n)
        # 5. Track best
        pass

    return best, best_sum

def run_packing():
    n = 26
    best_overall = None
    best_overall_sum = 0.0

    for seed in range(10):
        result, score = generate_candidates(n, seed * 13 + 42)
        if result is not None and score > best_overall_sum:
            best_overall = result
            best_overall_sum = score

    if best_overall is None:
        # Fallback: simple grid, guaranteed valid
        cols = 6
        centers = []
        for row in range(5):
            for col in range(cols):
                x = (col + 0.5) / cols
                y = (row + 0.5) / 5
                centers.append([x, y])
        centers = np.array(centers[:n], dtype=float)
        radii = np.full(n, min(0.5/cols, 0.5/5) * 0.9)
        centers, radii = repair(centers, radii, n)
        return centers, radii, float(np.sum(radii))

    centers, radii = best_overall
    return centers, radii, best_overall_sum
```

### CRITICAL DO-NOT-DO (top failures from real runs)

1. **MUST return EXACTLY n=26 circles.** Do NOT return 25, 27, or any other number.
   Before returning, verify: `assert centers.shape == (26, 2)`, `assert radii.shape == (26,)`.
2. **NEVER use nested function closures.** All helper functions must be defined at the
   top level (not inside `run_packing`). Nested closures cause `SyntaxError: name used
   prior to global declaration`.
3. **NEVER use `global` or `nonlocal` keywords inside nested functions.**
4. **ALWAYS call `repair()` before validating.** Unrepaired layouts are almost always
   invalid. Unrepaired is the #1 cause of overlap/out-of-bounds rejections.
5. **Do NOT redefine `validate` or `repair`.** Use the provided implementations.
   Self-written versions often miss edge cases (e.g., square boundary underflow).
6. **radii must be >= 0.** Call `np.maximum(radii, 0.001)` after any expansion step.

### Additional rules

- Always track the BEST valid result across trials. Return that, not the last trial.
- Use `np.random.RandomState(seed)` for reproducibility.
- The fallback grid must be valid. Test it mentally: is the min radius truly small enough?
- SLSQP and scipy.optimize are NOT recommended — they often produce invalid results. Use iterative repair + expansion instead.
