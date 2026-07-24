# Erdos Minimum Overlap

Find a step function h on [0, 2] minimizing the overlap integral C5.
h is represented as `n_points` samples on a uniform grid.

Constraints:
- 0 <= h[i] <= 1 for all i
- sum(h) = n_points / 2

The evaluator computes:
```python
C5 = max(np.correlate(h, 1 - h, mode="full") * (2.0 / n_points))
```

Goal: minimize C5. Reference is ~0.38 for n_points=80.

## REQUIRED interface

```python
def run(seed=42, budget_s=20, **kwargs):
    """Returns (h_values, c5_bound, n_points)."""
    # h_values: (n_points,) float ndarray in [0,1], sum = n_points/2
    # c5_bound: float — the C5 value your construction achieves
    # n_points: int — the number of discretization points used
    return h_values, c5_bound, n_points
```

A global `initial_h_values` (numpy array, in [0,1], sum already balanced) is available.
Use it as a starting point if helpful, but you may ignore it.

## Reliable utilities — copy these into your code exactly

### Compute C5

```python
def compute_c5(h, n_points):
    dx = 2.0 / n_points
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    correlation = np.correlate(h, one_minus_h, mode="full") * dx
    return float(np.max(correlation))
```

### Normalize to feasible region

After any modification to h, call this to re-establish constraints:

```python
def normalize(h, n_points):
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    target_sum = n_points / 2.0
    current_sum = np.sum(h)
    if current_sum > 1e-12:
        h = h * (target_sum / current_sum)
    else:
        h = np.full(n_points, 0.5)
    h = np.clip(h, 0.0, 1.0)
    target_sum = n_points / 2.0
    excess = np.sum(h) - target_sum
    if abs(excess) > 0.01:
        h = h * (target_sum / np.sum(h))
    return h
```

### Apply a local move that preserves the sum

Move `delta` mass from index `src` to index `dst`, then normalize:

```python
def move_mass(h, src, dst, delta):
    h = np.asarray(h, dtype=float).copy()
    h[src] = max(0.0, h[src] - delta)
    h[dst] = min(1.0, h[dst] + delta)
    return h
```

## Strategy — you fill in this function

Your job is to write `run()`. The utilities above are reliable — include and call them.

Use this structure:

```python
def run(seed=42, budget_s=20, **kwargs):
    n_points = 80
    rng = np.random.RandomState(seed)

    # Start from initial_h_values if available, else random feasible vector
    try:
        h = initial_h_values.copy().astype(float)
    except NameError:
        h = np.full(n_points, 0.5)
        h = normalize(h, n_points)

    best_h = h.copy()
    best_c5 = compute_c5(h, n_points)
    start_time = time.time()
    time_limit = budget_s * 0.95

    # === YOUR STRATEGY HERE ===
    # Iteratively refine h to minimize C5. Try:
    #
    # 1. Local mass transfer: pick two indices, move a small amount from
    #    one to the other, accept if C5 improves.
    # 2. Symmetry breaking: try asymmetric structures. Pure constant h=0.5
    #    gives a specific C5; structured patterns often do better.
    # 3. Perturb-and-filter: add small noise, normalize, keep if better.
    # 4. Block smoothing: average h over windows of 3-5 indices.
    #
    # Key insight: C5 measures max overlap between h and (1-h) across all
    # shifts. Reducing this maximum requires making h less correlated with
    # its own shifted complement — which means avoiding periodic or
    # symmetric patterns.

    while time.time() - start_time < time_limit:
        # 1. Try a local modification of h
        # 2. Normalize to maintain constraints
        # 3. Compute new C5
        # 4. Accept if lower (or accept small increases occasionally
        #    for temperature/annealing early on)
        pass

    return best_h, best_c5, n_points
```

### CRITICAL DO-NOT-DO (top failures from real runs)

1. **ALWAYS call `normalize()` after ANY modification to h.** Skipping normalization is
   the #1 cause of rejection: h values leak outside [0,1] after clipping or scaling.
   Every perturbation block MUST end with `h = normalize(h, n_points)`.
2. **Report the EXACT c5_bound that `compute_c5()` returns.** Do NOT round, adjust, or
   re-compute C5 manually. Mismatch between reported and actual C5 is the #2 cause
   of rejection. Do: `c5 = compute_c5(h, n_points); return h, c5, n_points`.
3. **Do NOT use nested function closures.** Define all helper functions at top level.
   Nested closures cause `SyntaxError: name used prior to global declaration`.
4. **Do NOT use `global` or `nonlocal` keywords inside nested functions.**
5. **Do NOT redefine `compute_c5()` or `normalize()`.** Use the provided implementations.
   Self-written versions often forget to multiply by `dx` (2/n_points) or mishandle
   the double-clip+scale normalization.

### Additional rules

- Use `np.random.RandomState(seed)` for reproducibility.
- Budget your time: check `time.time()` periodically and return the best found before timeout.
- The improvement from 0.5 to 0.38 is about a 24% reduction. Most of it comes from breaking the symmetry of uniform h.
- CVXPY and scipy.optimize are NOT recommended — they often fail silently. Use iterative local search instead.
