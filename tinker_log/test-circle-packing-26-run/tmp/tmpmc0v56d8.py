import numpy as np
import math
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
# Simple stochastic hill‑climbing that moves only the centres.
# Radii are recomputed by the LP at every trial.
# ----------------------------------------------------------------------
def _local_search(centres_start: np.ndarray,
                  rng: np.random.Generator,
                  max_iters: int = 3000,
                  init_eps: float = 0.02,
                  decay: float = 0.995) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Perform a cheap random walk in centre space.
    Each proposal is evaluated by the LP defined above.
    """
    best_c = centres_start.copy()
    best_r = _refine_radii(best_c)
    best_total = float(best_r.sum())

    eps = init_eps

    for it in range(max_iters):
        # random jitter
        proposal = best_c + rng.uniform(-eps, eps, size=best_c.shape)
        proposal = np.clip(proposal, 0.0, 1.0)

        rad = _refine_radii(proposal)
        total = float(rad.sum())

        if total > best_total + 1e-12:
            best_c = proposal
            best_r = rad
            best_total = total
            # once an improvement is found, shrink step size for finer search
            eps = max(eps * 0.9, 1e-5)
        else:
            eps *= decay
            if eps < 1e-6:
                break

    return best_c, best_r, best_total


# ----------------------------------------------------------------------
# Extended hill‑climbing (more iterations, slower decay) – still centre only
# ----------------------------------------------------------------------
def _extended_local_search(centres_start: np.ndarray,
                           rng: np.random.Generator,
                           max_iters: int = 60000,
                           init_eps: float = 0.03,
                           decay: float = 0.998) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Same algorithm as _local_search but with many more iterations
    and a slower step‑size decay, enabling the optimiser to crawl
    into a higher‑quality region.
    """
    return _local_search(centres_start, rng,
                         max_iters=max_iters,
                         init_eps=init_eps,
                         decay=decay)


# ----------------------------------------------------------------------
# Main entry point required by the problem statement
# ----------------------------------------------------------------------
def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a feasible packing of 26 circles in the unit square that
    maximises the sum of radii.
    """
    n = 26
    target = 2.636  # required sum of radii

    rng = np.random.default_rng(seed=0)      # deterministic RNG
    base = _initial_guess_hex()

    best_total = -np.inf
    best_centres = None
    best_radii = None

    # ------------------------------------------------------------------
    # Stage 1 – many SLSQP restarts (exploration of the non‑convex space)
    # ------------------------------------------------------------------
    n_starts = 300               # more restarts than the original code
    for k in range(n_starts):
        # Every seventh start is completely random to increase diversity,
        # the others are jittered versions of the hexagonal seed.
        if k == 0:
            start = base
        elif k % 7 == 0:
            start = _random_guess(rng)
        else:
            start = _perturbed_guess(base, rng,
                                     cen_scale=0.06,
                                     rad_scale=0.06)

        sol = _solve_one_start(start)

        centres = sol[:52].reshape((n, 2))
        radii = _refine_radii(centres)
        total = float(radii.sum())

        if total > best_total:
            best_total = total
            best_centres = centres.copy()
            best_radii = radii.copy()
            if best_total >= target:
                break   # early stop if target already reached

    # ------------------------------------------------------------------
    # Stage 2 – intensive centre‑only hill‑climbing (much longer)
    # ------------------------------------------------------------------
    if best_total < target:
        best_centres, best_radii, best_total = _extended_local_search(
            best_centres,
            rng,
            max_iters=60000,
            init_eps=0.03,
            decay=0.998
        )

    # ------------------------------------------------------------------
    # Stage 3 – final polishing with SLSQP (centres + radii together)
    # ------------------------------------------------------------------
    # Build a full vector from the best centres and radii found so far
    final_vec = np.empty(78)
    final_vec[:52] = best_centres.ravel()
    final_vec[52:] = best_radii
    final_vec = _solve_one_start(final_vec)

    # Re‑compute radii using the LP (to be completely safe)
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