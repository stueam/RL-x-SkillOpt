import numpy as np
from scipy.optimize import minimize, linprog
from typing import Tuple

# ----------------------------------------------------------------------
#  Constraint objects for SLSQP (must be top‑level)
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


# ----------------------------------------------------------------------
#  Helper to build the full list of SLSQP constraints
# ----------------------------------------------------------------------
def _make_constraints(n_circles: int):
    cons = []
    for i in range(n_circles):
        cons.append({'type': 'ineq', 'fun': InsideConstraint(i)})
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            cons.append({'type': 'ineq', 'fun': NonOverlapConstraint(i, j)})
    return cons


# ----------------------------------------------------------------------
#  Objective for SLSQP (maximise sum of radii → minimise negative sum)
# ----------------------------------------------------------------------
def _objective(x: np.ndarray) -> float:
    return -np.sum(x[52:])


# ----------------------------------------------------------------------
#  Initialise a reasonable hexagonal pattern (25 circles) + a centre one
# ----------------------------------------------------------------------
def _initial_guess_hex() -> np.ndarray:
    """
    Returns a feasible start vector (78 numbers).  The first 52 entries are
    (x, y) coordinates of 26 circles, the remaining 26 entries are their radii.
    """
    r0 = 0.09                     # base radius – comfortably fits the square
    dx = 2 * r0
    dy = np.sqrt(3) * r0

    rows = 5
    centres = []
    for row in range(rows):
        y = r0 + row * dy
        x_start = r0 + (row % 2) * r0   # offset every other row
        for k in range(5):
            x = x_start + k * dx
            centres.append([x, y])

    # extra circle roughly in the centre
    centres.append([0.5, 0.5])

    centres = np.array(centres)          # shape (26, 2)
    radii = np.full(26, r0)
    radii[-1] = 0.04                     # a little smaller for the middle one

    x0 = np.empty(78)
    x0[:52] = centres.ravel()
    x0[52:] = radii
    return x0


# ----------------------------------------------------------------------
#  Produce a completely random feasible start vector
# ----------------------------------------------------------------------
def _random_start(rng: np.random.Generator) -> np.ndarray:
    """
    Randomly place 26 points in the unit square and give them a tiny
    radius (0.02).  The generated vector is feasible for the SLSQP
    constraints because the radii are far smaller than any possible
    distance to a wall or to another centre.
    """
    centres = rng.uniform(0.0, 1.0, size=(26, 2))
    radii = np.full(26, 0.02)

    x0 = np.empty(78)
    x0[:52] = centres.ravel()
    x0[52:] = radii
    return x0


# ----------------------------------------------------------------------
#  Small perturbation of a given start (uniform jitter)
# ----------------------------------------------------------------------
def _perturbed_guess(base_x: np.ndarray,
                     rng: np.random.Generator,
                     cen_scale: float = 0.025,
                     rad_scale: float = 0.025) -> np.ndarray:
    """Return a copy of base_x with a small uniform random perturbation."""
    x = base_x.copy()
    centres = x[:52].reshape(-1, 2)
    radii = x[52:]

    centres += rng.uniform(-cen_scale, cen_scale, size=centres.shape)
    radii += rng.uniform(-rad_scale, rad_scale, size=radii.shape)
    radii = np.maximum(radii, 0.0)

    # clip centres to the unit square (still feasible because radii are tiny)
    centres = np.clip(centres, 0.0, 1.0)

    x[:52] = centres.ravel()
    x[52:] = radii
    return x


# ----------------------------------------------------------------------
#  One SLSQP optimisation starting from a given vector
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
        options={'ftol': 1e-12, 'maxiter': 3000, 'disp': False}
    )
    final = res.x if res.success else x0

    # force any tiny negative radii to zero (numerical noise)
    radii = final[52:]
    radii = np.where(radii < 0.0, 0.0, radii)
    final[52:] = radii
    return final


# ----------------------------------------------------------------------
#  Linear programme that maximises Σ r_i for *fixed* centres
# ----------------------------------------------------------------------
def _refine_radii(centres: np.ndarray) -> np.ndarray:
    """
    Given centre coordinates (shape (n,2)) return the optimal radii
    (shape (n,)) that maximise the sum of radii while respecting walls
    and pairwise non‑overlap.
    """
    n = centres.shape[0]

    rows = []
    b = []

    # wall constraints: r_i ≤ distance to each wall
    for i in range(n):
        xi, yi = centres[i]
        e = np.zeros(n)
        e[i] = 1.0
        rows.append(e.copy()); b.append(xi)            # left wall
        rows.append(e.copy()); b.append(1.0 - xi)      # right wall
        rows.append(e.copy()); b.append(yi)            # bottom wall
        rows.append(e.copy()); b.append(1.0 - yi)      # top wall

    # pairwise constraints: r_i + r_j ≤ distance(centre_i, centre_j)
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

    c = -np.ones(n)                     # maximise Σ r_i  → minimise -Σ r_i
    bounds = [(0.0, None)] * n          # radii are non‑negative

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        # Fallback – return zeros (should never happen)
        return np.zeros(n, dtype=float)
    return res.x


# ----------------------------------------------------------------------
#  Stochastic annealing that works directly on centre positions.
#  Radii are always recomputed by the optimal LP.
# ----------------------------------------------------------------------
def _anneal_on_centres(start_c: np.ndarray,
                       rng: np.random.Generator,
                       n_iters_phase1: int = 40000,
                       n_iters_phase2: int = 40000) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Two‑phase simulated annealing.
    * Phase 1 (coarse) explores the space with a large step size.
    * Phase 2 (fine) refines the best configuration found.
    Returns the best centre set, the associated radii and the sum of radii.
    """
    # -------------------- initialise --------------------
    cur_c = start_c.copy()
    cur_r = _refine_radii(cur_c)
    cur_total = float(cur_r.sum())

    best_c = cur_c.copy()
    best_r = cur_r.copy()
    best_total = cur_total

    # -------------- phase 1 – coarse annealing --------------
    sigma = 0.08          # large initial move size
    temp = 0.05           # relatively high temperature

    sigma_decay = (0.02 / sigma) ** (1.0 / n_iters_phase1)
    temp_decay = (0.01 / temp) ** (1.0 / n_iters_phase1)

    for it in range(n_iters_phase1):
        proposal = cur_c + rng.normal(0.0, sigma, size=cur_c.shape)
        proposal = np.clip(proposal, 0.0, 1.0)

        rad = _refine_radii(proposal)
        total = float(rad.sum())
        delta = total - cur_total

        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-12)):
            cur_c = proposal
            cur_r = rad
            cur_total = total
            if total > best_total:
                best_c = proposal.copy()
                best_r = rad.copy()
                best_total = total

        sigma *= sigma_decay
        temp *= temp_decay

        # occasional global jump to avoid pathological traps
        if (it + 1) % 5000 == 0:
            jump = _random_start(rng)[:52].reshape((26, 2))
            jump_r = _refine_radii(jump)
            jump_total = float(jump_r.sum())
            if jump_total > best_total:
                cur_c = jump.copy()
                cur_r = jump_r.copy()
                cur_total = jump_total
                best_c = jump.copy()
                best_r = jump_r.copy()
                best_total = jump_total

    # -------------- phase 2 – fine annealing --------------
    sigma = 0.02          # start fine‑grained moves
    temp = 0.01           # lower temperature

    sigma_decay = (0.001 / sigma) ** (1.0 / n_iters_phase2)
    temp_decay = (1e-6 / temp) ** (1.0 / n_iters_phase2)

    for it in range(n_iters_phase2):
        proposal = cur_c + rng.normal(0.0, sigma, size=cur_c.shape)
        proposal = np.clip(proposal, 0.0, 1.0)

        rad = _refine_radii(proposal)
        total = float(rad.sum())
        delta = total - cur_total

        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-12)):
            cur_c = proposal
            cur_r = rad
            cur_total = total
            if total > best_total:
                best_c = proposal.copy()
                best_r = rad.copy()
                best_total = total

        sigma *= sigma_decay
        temp *= temp_decay

    return best_c, best_r, best_total


# ----------------------------------------------------------------------
#  Main entry point required by the validator
# ----------------------------------------------------------------------
def run_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a feasible packing of 26 circles in the unit square that
    maximises the sum of radii.  The returned tuple is
        (centres, radii, sum_of_radii)
    where `centres` has shape (26,2) and `radii` has shape (26,).
    """
    rng = np.random.default_rng(seed=42)   # deterministic RNG

    # --------------------------------------------------------------
    # Stage A – many short SLSQP runs (exploration)
    # --------------------------------------------------------------
    n_circles = 26
    best_total = -np.inf
    best_centres = None
    best_radii = None

    # prepare a collection of start vectors:
    #   * one deterministic hexagonal layout,
    #   * several modestly perturbed copies of that layout,
    #   * a handful of completely random layouts.
    base = _initial_guess_hex()
    starts = [base]

    # modest perturbations of the hex layout
    for _ in range(30):
        starts.append(_perturbed_guess(base, rng, cen_scale=0.04, rad_scale=0.04))

    # a small set of completely random feasible starts
    for _ in range(10):
        starts.append(_random_start(rng))

    # run SLSQP on each start
    for start in starts:
        sol = _solve_one_start(start)

        centres = sol[:52].reshape((n_circles, 2))
        radii = _refine_radii(centres)
        total = float(radii.sum())

        if total > best_total:
            best_total = total
            best_centres = centres.copy()
            best_radii = radii.copy()
            # early exit if we already beat the target
            if best_total >= 2.636:
                break

    # --------------------------------------------------------------
    # Stage B – a larger pool of random SLSQP restarts (more exhaustive)
    # --------------------------------------------------------------
    # If we are still below the target we try many more random starts.
    if best_total < 2.636:
        for _ in range(360):                     # total SLSQP runs ≈ 400
            start = _random_start(rng)
            sol = _solve_one_start(start)

            centres = sol[:52].reshape((n_circles, 2))
            radii = _refine_radii(centres)
            total = float(radii.sum())

            if total > best_total:
                best_total = total
                best_centres = centres.copy()
                best_radii = radii.copy()
                if best_total >= 2.636:
                    break

    # --------------------------------------------------------------
    # Stage C – two‑phase simulated annealing on the centre positions
    # --------------------------------------------------------------
    if best_total < 2.636:
        best_centres, best_radii, best_total = _anneal_on_centres(
            best_centres,
            rng,
            n_iters_phase1=40000,
            n_iters_phase2=40000
        )

    # --------------------------------------------------------------
    # Final safety fallback (should never be needed)
    # --------------------------------------------------------------
    if best_centres is None:
        fallback = _initial_guess_hex()
        best_centres = fallback[:52].reshape((n_circles, 2))
        best_radii = fallback[52:]
        best_total = float(best_radii.sum())

    return best_centres, best_radii, best_total