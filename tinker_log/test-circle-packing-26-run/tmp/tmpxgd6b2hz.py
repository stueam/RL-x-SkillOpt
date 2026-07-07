# --------------------------------------------------------------
#  Circle‑packing for 26 circles in the unit square (enhanced)
# --------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize, linprog
from typing import Tuple

# ------------------------------------------------------------------
#  Constraint objects required by SLSQP (must be top‑level)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
#  Build the full list of inequality constraints for SLSQP
# ------------------------------------------------------------------
def _make_constraints(n_circles: int):
    cons = []
    # wall constraints
    for i in range(n_circles):
        cons.append({'type': 'ineq', 'fun': InsideConstraint(i)})
    # pairwise non‑overlap constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            cons.append({'type': 'ineq', 'fun': NonOverlapConstraint(i, j)})
    return cons


# ------------------------------------------------------------------
#  Objective for SLSQP (maximise sum of radii → minimise negative sum)
# ------------------------------------------------------------------
def _objective(x: np.ndarray) -> float:
    # x[52:] contains the radii
    return -np.sum(x[52:])


# ------------------------------------------------------------------
#  Helper: initialise a feasible hexagonal pattern (25 circles) + 1 centre
# ------------------------------------------------------------------
def _initial_guess_hex() -> np.ndarray:
    """
    Returns a feasible start vector (78 numbers).  The first 52 entries are
    (x, y) coordinates of 26 circles, the remaining 26 entries are their radii.
    """
    r0 = 0.09                     # base radius – comfortably fits the square
    dx = 2 * r0                   # horizontal spacing of triangular lattice
    dy = np.sqrt(3) * r0          # vertical spacing of triangular lattice

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


# ------------------------------------------------------------------
#  Perturb a start vector (uniform jitter)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
#  One SLSQP optimisation starting from a given vector
# ------------------------------------------------------------------
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
        options={'ftol': 1e-12, 'maxiter': 2500, 'disp': False}
    )
    final = res.x if res.success else x0

    # force any tiny negative radii to zero (numerical noise)
    radii = final[52:]
    radii = np.where(radii < 0.0, 0.0, radii)
    final[52:] = radii
    return final


# ------------------------------------------------------------------
#  Linear programme that maximises Σ r_i for *fixed* centres
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
#  Deterministic hill‑climbing on the centre positions (all at once)
# ------------------------------------------------------------------
def _local_search(centres_start: np.ndarray,
                  rng: np.random.Generator,
                  max_iters: int = 8000,
                  init_eps: float = 0.025,
                  decay: float = 0.996) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simple stochastic hill‑climbing that moves *all* centres at once.
    Radii are recomputed by the LP after each trial.
    """
    best_c = centres_start.copy()
    best_r = _refine_radii(best_c)
    best_total = float(best_r.sum())

    eps = init_eps

    for _ in range(max_iters):
        proposal = best_c + rng.uniform(-eps, eps, size=best_c.shape)
        proposal = np.clip(proposal, 0.0, 1.0)

        rad = _refine_radii(proposal)
        total = float(rad.sum())

        if total > best_total + 1e-9:
            best_c = proposal
            best_r = rad
            best_total = total
            eps = max(eps * 0.9, 1e-5)   # tighten step size after improvement
        else:
            eps *= decay
            if eps < 1e-6:
                break

    return best_c, best_r, best_total


# ------------------------------------------------------------------
#  Fine‑grained *single‑circle* hill‑climbing
# ------------------------------------------------------------------
def _single_circle_fine_tune(centres_start: np.ndarray,
                            rng: np.random.Generator,
                            max_iters: int = 30000,
                            init_eps: float = 0.02,
                            decay: float = 0.99) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Moves one randomly chosen circle at a time.
    Each trial recomputes the optimal radii via the LP.
    The step size decays gradually.
    """
    best_c = centres_start.copy()
    best_r = _refine_radii(best_c)
    best_total = float(best_r.sum())
    eps = init_eps

    n = best_c.shape[0]

    for _ in range(max_iters):
        i = rng.integers(n)                     # pick a random circle
        proposal = best_c.copy()
        proposal[i] += rng.uniform(-eps, eps, size=2)
        proposal[i] = np.clip(proposal[i], 0.0, 1.0)

        rad = _refine_radii(proposal)
        total = float(rad.sum())

        if total > best_total + 1e-9:
            best_c = proposal
            best_r = rad
            best_total = total
            eps = max(eps * 0.9, 1e-6)          # sharpen after a win
        else:
            eps *= decay
            if eps < 1e-6:
                break

    return best_c, best_r, best_total


# ------------------------------------------------------------------
#  Fine‑grained simulated annealing / long random walk (as in original)
# ------------------------------------------------------------------
def _simulated_annealing(centres_start: np.ndarray,
                         rng: np.random.Generator,
                         n_iters: int = 20000,
                         sigma_start: float = 0.03,
                         sigma_end: float = 0.001,
                         temp_start: float = 0.02,
                         temp_end: float = 5e-6) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    A greedy + occasional‑accept walk that slowly reduces the proposal width
    and the acceptance temperature.
    """
    best_c = centres_start.copy()
    best_r = _refine_radii(best_c)
    best_total = float(best_r.sum())

    cur_c = best_c.copy()
    cur_r = best_r.copy()
    cur_total = best_total

    sigma = sigma_start
    temp = temp_start
    # pre‑compute exponential decay factors
    sigma_decay = (sigma_end / sigma_start) ** (1.0 / n_iters)
    temp_decay = (temp_end / temp_start) ** (1.0 / n_iters)

    for _ in range(n_iters):
        proposal = cur_c + rng.normal(0.0, sigma, size=cur_c.shape)
        proposal = np.clip(proposal, 0.0, 1.0)

        rad = _refine_radii(proposal)
        total = float(rad.sum())
        delta = total - cur_total

        # Accept if better, otherwise with Metropolis probability
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-12)):
            cur_c = proposal
            cur_r = rad
            cur_total = total
            if total > best_total:
                best_c = proposal
                best_r = rad
                best_total = total

        sigma *= sigma_decay
        temp *= temp_decay

    return best_c, best_r, best_total


# ------------------------------------------------------------------
#  Main entry point
# ------------------------------------------------------------------
def run_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a feasible packing of 26 circles in the unit square that
    maximises the sum of radii.  The returned tuple is
        (centres, radii, sum_of_radii)
    where `centres` has shape (26,2) and `radii` has shape (26,).
    """
    n = 26
    target = 2.636           # we aim to beat this value

    rng = np.random.default_rng(seed=0)   # deterministic RNG
    base = _initial_guess_hex()

    best_total = -np.inf
    best_centres = None
    best_radii = None

    # --------------------------------------------------------------
    # Stage 1 – many short SLSQP runs (exploration)
    # --------------------------------------------------------------
    n_starts = 400               # increased number of restarts
    for k in range(n_starts):
        if k == 0:
            start = base
        else:
            start = _perturbed_guess(base, rng,
                                     cen_scale=0.03,
                                     rad_scale=0.03)

        sol = _solve_one_start(start)

        centres = sol[:52].reshape((n, 2))
        radii = _refine_radii(centres)
        total = float(radii.sum())

        if total > best_total:
            best_total = total
            best_centres = centres.copy()
            best_radii = radii.copy()
            if best_total >= target:
                break

    # --------------------------------------------------------------
    # Stage 2 – deterministic hill‑climbing on all centres
    # --------------------------------------------------------------
    if best_total < target:
        best_centres, best_radii, best_total = _local_search(
            best_centres,
            rng,
            max_iters=8000,
            init_eps=0.025,
            decay=0.996
        )

    # --------------------------------------------------------------
    # Stage 3 – single‑circle fine‑tuning (new phase)
    # --------------------------------------------------------------
    if best_total < target:
        best_centres, best_radii, best_total = _single_circle_fine_tune(
            best_centres,
            rng,
            max_iters=30000,
            init_eps=0.02,
            decay=0.99
        )

    # --------------------------------------------------------------
    # Stage 4 – fine‑grained simulated annealing / long random walk
    # --------------------------------------------------------------
    if best_total < target:
        best_centres, best_radii, best_total = _simulated_annealing(
            best_centres,
            rng,
            n_iters=20000,
            sigma_start=0.03,
            sigma_end=0.001,
            temp_start=0.02,
            temp_end=5e-6
        )

    # safety fallback (should never be needed)
    if best_centres is None:
        flat = base
        best_centres = flat[:52].reshape((n, 2))
        best_radii = flat[52:]
        best_total = float(best_radii.sum())

    return best_centres, best_radii, best_total