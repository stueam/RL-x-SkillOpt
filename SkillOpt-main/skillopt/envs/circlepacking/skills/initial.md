# Circle Packing Strategies

## Problem
Pack N circles in a unit square [0,1]x[0,1] to maximize sum of radii.
Constraints: no overlaps, all circles inside square, radii non-negative.

## Basic Strategy

### Hexagonal (Triangular) Lattice
The densest packing of equal circles in the plane. Use this as a starting
point: place circles in staggered rows. For unequal circles, start with
a hexagonal backbone and let optimization adjust sizes.

### Corner and Edge Placement
Large circles benefit from corners and edges because the square boundaries
constrain them from two or one sides, effectively giving them more room.
Place the largest circles in corners first.

### Greedy Construction
1. Place circles one by one, largest first, at feasible positions.
2. For each new circle, scan the available free space and place it where
   it fits without overlapping existing circles.
3. Fill corners and edges before the interior.

### Local Optimization with SLSQP
After initial placement, refine with scipy.optimize.minimize:
- Flatten variables as [x1..xN, y1..yN, r1..rN] (2N + N = 3N total).
- Objective: negative sum of radii (we want to maximize sum).
- Define a single constraints() function returning an array of all
  inequality constraints (each must be >= 0).
  - Boundary: for each circle i: x_i - r_i >= 0, y_i - r_i >= 0,
    1 - (x_i + r_i) >= 0, 1 - (y_i + r_i) >= 0.
  - Non-overlap: for each pair (i,j): dist^2 - (r_i + r_j)^2 >= 0.
- Pass as constraints={'type': 'ineq', 'fun': constraints}.
- Use options maxiter=2000-5000, ftol=1e-9 to 1e-12.

### Penalty Formulation (Alternative)
An alternative to hard constraints: encode violations as penalties
added to the objective:
  penalty = w1 * sum(boundary_violations^2) + w2 * sum(overlap_violations^2)
  objective = -sum(radii) + penalty
This is more forgiving of constraint violations during optimization.

## Critical Implementation Rules

### Variable Layout
Use flattened array: v[:N]=x coords, v[N:2N]=y coords, v[2N:]=radii.
This avoids indexing confusion and simplifies constraint functions.

### Constraint Function Pattern (RECOMMENDED)
```python
def constraints(v):
    cx, cy, cr = v[:N], v[N:2*N], v[2*N:]
    cons = list(cx - cr)           # left boundary
    cons.extend(cy - cr)           # bottom boundary
    cons.extend(1.0 - cx - cr)     # right boundary
    cons.extend(1.0 - cy - cr)     # top boundary
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            cons.append(dx*dx + dy*dy - (cr[i] + cr[j])**2)
    return np.array(cons)
```

### Post-Processing (CRITICAL)
After optimization, always add:
```python
radii = np.maximum(radii, 0.0)  # ensure non-negative
```
Then for strict feasibility, cap radii to respect boundaries and
iteratively shrink overlapping pairs:
```python
for i in range(N):
    max_r = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
    if radii[i] > max_r:
        radii[i] = max(0.001, max_r)
for _ in range(100):
    fixed = True
    for i in range(N):
        for j in range(i+1, N):
            d = sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            if d < radii[i] + radii[j]:
                overlap = (radii[i] + radii[j] - d) / 2
                radii[i] = max(0.001, radii[i] - overlap)
                radii[j] = max(0.001, radii[j] - overlap)
                fixed = False
    if fixed: break
```

### Common Bugs to AVOID
1. NEVER subtract a scalar from a list: `[a, b, c] - eps` is a TypeError.
   Use `min(a, b, c) - eps` instead of `min([a, b, c]) - eps`.
2. NEVER use lambda closures in loops without default arguments,
   or all closures capture the final value.
3. NEVER define run_packing with nested functions that use closures.
   Define helper functions at top level.
4. Always call `np.random.seed(seed)` at the start of run_packing()
   to ensure reproducibility.
5. The optimizer may return slightly infeasible solutions. Always
   include post-processing to enforce constraints strictly.

## Implementation
- Use `np.random.seed(seed)` for reproducibility.
- Define `run_packing()` returning (centers, radii, sum_radii).
- Use numpy arrays, scipy.optimize.minimize.
- No file or network I/O.
- Think step by step, output code between ```python and ```.
