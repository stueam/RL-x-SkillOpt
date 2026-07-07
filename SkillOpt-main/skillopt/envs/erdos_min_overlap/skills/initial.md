# Erdos Minimum Overlap Problem Strategies

## Problem
Find a step function h: [0, 2] → [0, 1] that minimizes the overlap integral:
C5 = max_k ∫ h(x)(1 - h(x+k)) dx

Constraints:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

## Basic Strategy

### Discretization
Represent h as n_points samples on a uniform grid over [0, 2):
- h[i] = h(i * dx) where dx = 2.0 / n_points
- Constraint: sum(h) * dx = 1, i.e. sum(h) = n_points / 2

### Smooth Initialization
Start with a smooth function like:
- Constant: h[i] = 0.5 for all i (satisfies constraint for any n_points)
- Sinusoidal: h[i] = 0.5 + A * sin(2π * i / n_points + φ) (requires normalization)
- Step-like: piecewise constant with random perturbations

Then optimize to minimize C5.

### Convex Optimization with CVXPY
The problem can be formulated as a convex optimization:
- Variables: h[0], ..., h[n_points-1]
- Objective: minimize the maximum correlation
- Constraints: 0 ≤ h[i] ≤ 1, sum(h) = n_points / 2

```python
import cvxpy as cp
h = cp.Variable(n_points)
constraints = [0 <= h, h <= 1, cp.sum(h) == n_points / 2]
```

For the max-over-correlation objective, introduce auxiliary variable t:
```python
t = cp.Variable()
j = 1 - h
corr = cp.conv(h, j)  # This gives correlation for all shifts
objective = cp.Minimize(t)
constraints += [corr * dx <= t]
```

### SLSQP / Gradient-Based Optimization
Alternative to CVXPY using scipy.optimize.minimize:
- Flatten variables as a 1D array h[0..n-1]
- Objective: C5 = max correlation
- Constraints: sum(h) == n_points / 2, bounds [0, 1]

### Iterative Shrinking
1. Start with a feasible initial h (e.g., all 0.5).
2. Compute C5 and identify which shifts k give the largest overlap.
3. Locally adjust h to reduce those specific correlations.
4. Repeat until convergence.

## Critical Implementation Rules

### Normalization
After any update, ALWAYS renormalize:
```python
h = np.clip(h, 0.0, 1.0)
target_sum = n_points / 2.0
h = h * (target_sum / np.sum(h))
```

### Computing C5 Correctly
```python
dx = 2.0 / n_points
j = 1.0 - h
correlation = np.correlate(h, j, mode="full") * dx
c5 = np.max(correlation)
```

### Common Bugs to AVOID
1. NOT normalizing sum(h) to n_points/2 after optimization.
2. Using h values outside [0, 1] range.
3. Forgetting dx scaling: correlation * dx, not raw correlation.
4. Reporting a different c5_bound than what the actual computation yields.
5. Using nested functions with closures — define helpers at top level.

## Implementation
- Use `np.random.seed(seed)` for reproducibility.
- Define `run(seed=42)` returning (h_values, c5_bound, n_points).
- Use numpy arrays, scipy.optimize.minimize, cvxpy.
- No file or network I/O.
- Think step by step, output code between ```python and ```.
