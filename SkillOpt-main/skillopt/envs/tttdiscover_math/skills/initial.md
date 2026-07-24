# TTT-Discover Math Skill

Use the skill to generate correct, executable Python code.
Always return one code block with the exact entrypoint requested by the task.

## Shared Rules

- Use only numpy, math, random, itertools, time. No scipy, cvxpy, or external libraries.
- Track the BEST valid result across all trials. Return the best, not the last.
- Include a deterministic fallback that guarantees a valid output.
- Validate internally before returning. Invalid output scores zero.
- Prefer iterative local search (repair + grow / perturb + accept) over black-box optimizers.

### CRITICAL: Top failure patterns to avoid

1. **NEVER use nested function closures.** All helper functions MUST be defined at the
   top level of the module (not inside `run_packing` or `run`). Nested closures cause
   `SyntaxError: name used prior to global declaration` on this model.
2. **NEVER use `global` or `nonlocal` inside functions.**
3. **Do NOT redefine the provided utilities** — use `validate`, `repair`, `compute_c5`,
   `normalize` exactly as given. Self-written versions often have bugs in boundary math
   or normalization logic.

## Circle Packing 26

Objective: maximize `sum(radii)` for 26 non-overlapping circles in [0,1]x[0,1].

### Reliable utilities (copy exactly)

```python
def validate(centers, radii, n):
    if np.any(radii < 0): return False, 0.0
    if np.any(centers < 0) or np.any(centers > 1): return False, 0.0
    for i in range(n):
        if centers[i,0]-radii[i]<-1e-9 or centers[i,1]-radii[i]<-1e-9: return False, 0.0
        if centers[i,0]+radii[i]>1+1e-9 or centers[i,1]+radii[i]>1+1e-9: return False, 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx=centers[i,0]-centers[j,0]; dy=centers[i,1]-centers[j,1]
            if np.sqrt(dx*dx+dy*dy) < radii[i]+radii[j]-1e-9: return False, 0.0
    return True, float(np.sum(radii))

def repair(centers, radii, n):
    radii = np.maximum(radii, 0.001)
    for _ in range(200):
        fixed = True
        for i in range(n):
            for j in range(i+1, n):
                dx=centers[i,0]-centers[j,0]; dy=centers[i,1]-centers[j,1]
                dist=np.sqrt(dx*dx+dy*dy)
                if dist<radii[i]+radii[j]-1e-9:
                    overlap=(radii[i]+radii[j]-dist)/2
                    radii[i]=max(0.001,radii[i]-overlap*0.5)
                    radii[j]=max(0.001,radii[j]-overlap*0.5)
                    fixed=False
        if fixed: break
    for i in range(n):
        max_r=min(centers[i,0],1-centers[i,0],centers[i,1],1-centers[i,1])
        radii[i]=min(radii[i],max_r)
    return centers, radii
```

### Your job

Implement `run_packing()` returning `(centers, radii, sum_radii)`.
- centers: (26,2) float ndarray — **must be exactly 26, never 25 or 27**
- radii: (26,) float ndarray — **always >= 0, call np.maximum(radii, 0.001) before returning**
- Before returning: assert `centers.shape == (26, 2)` and `radii.shape == (26,)`

Strategy: multi-trial search with different layouts, repair each, keep best.
Try hexagonal grids, corner-priority placements, greedy fills, jittered voronoi.
Fallback: 6x5 grid with radii = min(1/12, 1/10) * 0.9, then repair.

## Erdos Minimum Overlap

Objective: minimize C5 = max(np.correlate(h, 1-h)) * 2/n_points.
Constraint: 0<=h[i]<=1, sum(h)=n_points/2.

### Reliable utilities (copy exactly)

```python
def compute_c5(h, n_points):
    dx = 2.0 / n_points
    return float(np.max(np.correlate(h, 1.0-h, mode="full") * dx))

def normalize(h, n_points):
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if s > 1e-12:
        h = h * (n_points/2.0 / s)
    else:
        h = np.full(n_points, 0.5)
    return np.clip(h, 0.0, 1.0)
```

### Your job

Implement `def run(seed=42, budget_s=20, **kwargs):` returning `(h_values, c5_bound, n_points)`.
- A global `initial_h_values` may be available.
- **After EVERY modification, call `h = normalize(h, n_points)`.**
- **Return the EXACT c5 from `compute_c5()`, never round or adjust.**

Strategy: start from initial_h_values or uniform 0.5. Use local mass moves
(pick two indices, transfer small amount) + normalize, accept if C5 improves.
Add small perturbations, try block smoothing over windows of 3-5 indices.
Break symmetry — uniform h=0.5 gives moderate C5; structured patterns do better.
Use time.time() to budget within budget_s seconds. Report the actual computed C5.
