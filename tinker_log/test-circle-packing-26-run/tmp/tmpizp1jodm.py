import numpy as np
from scipy.optimize import minimize, linprog

# ----------------------------------------------------------------------
# Constraint objects for SLSQP (must be top‑level, no closures)
# ----------------------------------------------------------------------
class InsideConstraint:
    """Four wall constraints for circle i: left, right, bottom, top."""
    def __init__(self, i: int):
        self.i = i

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xi = x[2 * self.i]
        yi = x[2 * self.i + 1]
        ri = x[52 + self.i]
        return np.array([
            xi - ri,               # left wall
            1.0 - xi - ri,         # right wall
            yi - ri,               # bottom wall
            1.0 - yi - ri          # top wall
        ])


class NonOverlapConstraint:
    """Non‑overlap constraint for circle pair (i, j)."""
    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j

    def __call__(self, x: np.ndarray) -> float:
        xi = x[2 * self.i]
        yi = x[2 * self.i + 1]
        xj = x[2 * self.j]
        yj = x[2 * self.j + 1]
        ri = x[52 + self.i]
        rj = x[52 + self.j]
        return np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) - (ri + rj)


def _make_constraints(n_circles: int):
    """Build the full list of inequality constraints required by SLSQP."""
    cons = []
    # wall constraints
    for i in range(n_circles):
        cons.append({'type': 'ineq', 'fun': InsideConstraint(i)})
    # pairwise non‑overlap constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            cons.append({'type': 'ineq', 'fun': NonOverlapConstraint(i, j)})
    return cons


# ----------------------------------------------------------------------
# Objective for SLSQP (maximise sum of radii → minimise negative sum)
# ----------------------------------------------------------------------
def _objective(x: np.ndarray) -> float:
    return -np.sum(x[52:])


# ----------------------------------------------------------------------
# Helper to build the initial hexagonal layout (25 points + 1 centre)
# ----------------------------------------------------------------------
def _initial_guess_hex() -> np.ndarray:
    """
    Returns a feasible start vector (78 numbers) in which the first 52
    entries are (x, y) coordinates of 26 circles and the remaining 26
    entries are their radii.
    """
    r0 = 0.09                      # small feasible radius
    dx = 2 * r0                     # horizontal spacing in a triangular lattice
    dy = np.sqrt(3) * r0           # vertical spacing

    rows = 5
    centres = []
    for row in range(rows):
        y = r0 + row * dy
        x_start = r0 + (row % 2) * r0   # offset every other row
        for k in range(5):
            x = x_start + k * dx
            centres.append([x, y])

    # add a single extra circle at the centre of the square
    centres.append([0.5, 0.5])

    centres = np.array(centres)           # (26, 2)
    radii = np.full(26, r0)
    radii[-1] = 0.04                      # a little smaller for the extra centre circle

    x0 = np.empty(78)
    x0[:52] = centres.ravel()
    x0[52:] = radii
    return x0


# ----------------------------------------------------------------------
# Perturb a start vector (uniform jitter)
# ----------------------------------------------------------------------
def _perturbed_guess(base_x: np.ndarray,
                     rng: np.random.Generator,
                     cen_scale: float = 0.025,
                     rad_scale: float = 0.025) -> np.ndarray:
    """Return a copy of base_x with a uniform random perturbation."""
    x = base_x.copy()
    centres = x[:52].reshape(-1, 2)
    radii = x[52:]

    centres += rng.uniform(-cen_scale, cen_scale, size=centres.shape)
    radii += rng.uniform(-rad_scale, rad_scale, size=radii.shape)
    radii = np.maximum(radii, 0.0)

    # keep centres inside the square – SLSQP will still be able to move them
    centres = np.clip(centres, 0.0, 1.0)

    x[:52] = centres.ravel()
    x[52:] = radii
    return x


# ----------------------------------------------------------------------
# Create a completely random start vector (useful for diversification)
# ----------------------------------------------------------------------
def _random_guess(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a feasible (but not necessarily optimal) random start vector.
    Radii are deliberately kept small to stay inside the feasible region.
    """
    n = 26
    centres = rng.uniform(0.0, 1.0, size=(n, 2))
    radii = rng.uniform(0.0, 0.06, size=n)   # modest radii – well within the square
    x0 = np.empty(78)
    x0[:52] = centres.ravel()
    x0[52:] = radii
    return x0


# ----------------------------------------------------------------------
# Run one SLSQP optimisation starting from x0
# ----------------------------------------------------------------------
def _solve_one_start(x0: np.ndarray) -> np.ndarray:
    n = 26
    constraints = _make_constraints(n)
    bounds = [(None, None)] * 52 + [(0.0, None)] * 26

    res = minimize(
        fun=_objective,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 4000, 'disp': False}
    )

    final = res.x if res.success else x0

    # remove tiny negatives caused by numerical noise
    radii = final[52:]
    radii = np.where(radii < 0.0, 0.0, radii)
    final[52:] = radii
    return final


# ----------------------------------------------------------------------
# Linear‑programme that maximises sum of radii for *fixed* centres
# ----------------------------------------------------------------------
def _refine_radii(centres: np.ndarray) -> np.ndarray:
    """
    LP: maximise Σ r_i
    subject to
        r_i ≤ x_i, 1‑x_i, y_i, 1‑y_i            (wall constraints)
        r_i + r_j ≤ dist(centre_i, centre_j)    (pairwise)
    """
    n = centres.shape[0]

    rows = []
    b = []

    # wall constraints
    for i in range(n):
        xi, yi = centres[i]
        e = np.zeros(n)

        e[i] = 1.0
        rows.append(e.copy()); b.append(xi)            # left
        rows.append(e.copy()); b.append(1.0 - xi)      # right
        rows.append(e.copy()); b.append(yi)            # bottom
        rows.append(e.copy()); b.append(1.0 - yi)      # top

    # pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(centres[i] - centres[j])
            e = np.zeros(n)
            e[i] = 1.0
            e[j] = 1.0
            rows.append(e)
            b.append(dij)

    A_ub = np.vstack(rows)
    b_ub = np.array(b)

    c = -np.ones(n)                     # maximise Σ r_i → minimise −Σ r_i
    bounds = [(0.0, None)] * n          # r_i ≥ 0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        # fallback – return zeros (should never happen)
        return np.zeros(n, dtype=float)
    return res.x


# ----------------------------------------------------------------------
# Greedy local search on centre positions
# ----------------------------------------------------------------------
def _greedy_local_search(centres_start: np.ndarray,
                         rng: np.random.Generator,
                         init_step: float = 0.025,
                         max_passes: int = 30) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simple deterministic pattern‑search:
    – for each circle try the eight directions (±step, ±step)
    – accept a move only if the total sum of radii improves
    – when a whole pass yields no improvement, halve the step.
    """
    centres = centres_start.copy()
    radii = _refine_radii(centres)
    best_total = float(radii.sum())
    step = init_step

    n = centres.shape[0]

    for _ in range(max_passes):
        improved = False
        order = rng.permutation(n)
        for i in order:
            # try the eight neighbour moves
            for dx, dy in (
                ( step,  0.0), (-step,  0.0),
                ( 0.0,  step), ( 0.0, -step),
                ( step,  step), ( step, -step),
                (-step,  step), (-step, -step)
            ):
                proposal = centres.copy()
                proposal[i, 0] = np.clip(proposal[i, 0] + dx, 0.0, 1.0)
                proposal[i, 1] = np.clip(proposal[i, 1] + dy, 0.0, 1.0)

                new_radii = _refine_radii(proposal)
                new_total = float(new_radii.sum())

                if new_total > best_total + 1e-12:
                    centres = proposal
                    radii = new_radii
                    best_total = new_total
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
    return centres, radii, best_total


# ----------------------------------------------------------------------
# Main entry point required by the problem statement
# ----------------------------------------------------------------------
def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a feasible packing of 26 circles in the unit square that
    maximises the sum of radii.
    Returns (centres, radii, total_sum_of_radii).
    """
    n = 26
    target = 2.636  # required sum of radii

    rng = np.random.default_rng(seed=0)      # deterministic RNG

    # ------------------------------------------------------------------
    # Stage 1 – many SLSQP restarts (exploration of the non‑convex space)
    # ------------------------------------------------------------------
    base = _initial_guess_hex()

    best_total = -np.inf
    best_centres = None
    best_radii = None

    n_starts = 400        # more restarts than the original version
    for k in range(n_starts):
        if k == 0:
            start = base
        elif k % 5 == 0:
            start = _random_guess(rng)
        else:
            start = _perturbed_guess(base, rng,
                                     cen_scale=0.04,
                                     rad_scale=0.04)

        sol = _solve_one_start(start)

        centres = sol[:52].reshape((n, 2))
        radii = _refine_radii(centres)
        total = float(radii.sum())

        if total > best_total:
            best_total = total
            best_centres = centres.copy()
            best_radii = radii.copy()
            if best_total >= target:
                break   # early exit if we already reached the target

    # ------------------------------------------------------------------
    # Stage 2 – greedy local search on centre positions only
    # ------------------------------------------------------------------
    if best_total < target:
        best_centres, best_radii, best_total = _greedy_local_search(
            best_centres,
            rng,
            init_step=0.03,
            max_passes=30
        )

    # ------------------------------------------------------------------
    # Stage 3 – final polishing with SLSQP (centres + radii together)
    # ------------------------------------------------------------------
    final_vec = np.empty(78)
    final_vec[:52] = best_centres.ravel()
    final_vec[52:] = best_radii
    final_vec = _solve_one_start(final_vec)

    best_centres = final_vec[:52].reshape((n, 2))
    best_radii = _refine_radii(best_centres)
    best_total = float(best_radii.sum())

    # ------------------------------------------------------------------
    # Safety fallback (should never be needed)
    # ------------------------------------------------------------------
    if best_centres is None:
        flat = base
        best_centres = flat[:52].reshape((n, 2))
        best_radii = flat[52:]
        best_total = float(best_radii.sum())

    return best_centres, best_radii, best_total