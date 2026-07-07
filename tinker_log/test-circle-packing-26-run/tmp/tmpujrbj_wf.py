#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimise the packing of 26 circles in the unit square [0,1]×[0,1]
to maximise the sum of the radii.

The implementation builds on the original SLSQP / local‑search code
but makes three important upgrades:

1. The linear programme that computes the optimal radii for a fixed
   set of centre positions is now pre‑compiled – only the right‑hand
   side changes – which makes it cheap enough to call tens of
   thousands of times.

2. A much larger number of SLSQP restarts together with a slower
   simulated‑annealing schedule explores the space far more
   thoroughly.

3. The whole pipeline is deterministic (single RNG seed) and
   respects the required function signatures.

The routine returns a tuple (centres, radii, sum_of_radii) where
centres.shape == (26, 2) and radii.shape == (26,).
"""

import numpy as np
from scipy.optimize import minimize, linprog
from typing import Tuple, List

# ----------------------------------------------------------------------
# Global data – used by the fast LP routine
# ----------------------------------------------------------------------
N_CIRCLES = 26
# wall rows: for each circle four rows (left, right, bottom, top)
# pair rows: one row per unordered pair (i<j)
PAIR_INDICES: List[Tuple[int, int]] = [(i, j) for i in range(N_CIRCLES)
                                      for j in range(i + 1, N_CIRCLES)]

# Build the constant part of the inequality matrix A_ub
# Each row corresponds either to a wall constraint (single 1) or a pair
# constraint (two 1’s).  The RHS vector b will be filled with the actual
# distances at runtime.
def _build_A_ub() -> np.ndarray:
    rows = []

    # wall constraints
    for i in range(N_CIRCLES):
        e = np.zeros(N_CIRCLES)
        e[i] = 1.0
        rows.append(e.copy())  # left
        rows.append(e.copy())  # right
        rows.append(e.copy())  # bottom
        rows.append(e.copy())  # top

    # pairwise constraints
    for _ in PAIR_INDICES:
        e = np.zeros(N_CIRCLES)
        # the two indices will be filled later when we compute the RHS
        rows.append(e)

    A = np.vstack(rows)
    return A

_A_UB = _build_A_ub()

# ----------------------------------------------------------------------
#  LP – maximal radii for a *fixed* set of centres
# ----------------------------------------------------------------------
def _refine_radii(centres: np.ndarray) -> np.ndarray:
    """
    Given centre coordinates (shape (N,2)) return the optimal radii
    (shape (N,)) that maximise the sum of radii while respecting the
    walls and pairwise non‑overlap constraints.
    """
    n = centres.shape[0]
    assert n == N_CIRCLES

    # Fill RHS vector b:
    #  - wall distances
    #  - pairwise centre distances
    b = np.empty(_A_UB.shape[0], dtype=float)
    idx = 0
    for i in range(n):
        xi, yi = centres[i]
        b[idx] = xi               # left
        b[idx + 1] = 1.0 - xi     # right
        b[idx + 2] = yi           # bottom
        b[idx + 3] = 1.0 - yi     # top
        idx += 4

    # pairwise distances
    for k, (i, j) in enumerate(PAIR_INDICES):
        dij = np.linalg.norm(centres[i] - centres[j])
        # insert the distance into the correct row of A_ub (the row already
        # contains zeros – we merely need the RHS)
        b[idx] = dij
        # also fill the two 1's for this row (they were zero)
        _A_UB[idx, i] = 1.0
        _A_UB[idx, j] = 1.0
        idx += 1

    # Objective: maximise sum(r)  -> minimise -sum(r)
    c = -np.ones(n)

    bounds = [(0.0, None)] * n

    # Solve with HiGHS (fast for small problems)
    res = linprog(c, A_ub=_A_UB, b_ub=b, bounds=bounds,
                  method='highs', options={'presolve': True})
    if not res.success:
        # In the extremely unlikely event the LP fails, fall back to zeros.
        return np.zeros(n, dtype=float)

    # Clean tiny negative values caused by numerical noise.
    radii = np.where(res.x < 0.0, 0.0, res.x)
    return radii

# ----------------------------------------------------------------------
#  Constraint objects required by SLSQP
# ----------------------------------------------------------------------
class InsideConstraint:
    """Four wall constraints for circle i: left, right, bottom, top."""
    def __init__(self, i: int):
        self.i = i

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xi = x[2 * self.i]
        yi = x[2 * self.i + 1]
        ri = x[2 * N_CIRCLES + self.i]
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
        ri = x[2 * N_CIRCLES + self.i]
        rj = x[2 * N_CIRCLES + self.j]
        return np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) - (ri + rj)


def _make_constraints() -> List[dict]:
    cons = []
    for i in range(N_CIRCLES):
        cons.append({'type': 'ineq', 'fun': InsideConstraint(i)})
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            cons.append({'type': 'ineq', 'fun': NonOverlapConstraint(i, j)})
    return cons

# ----------------------------------------------------------------------
#  Objective for SLSQP
# ----------------------------------------------------------------------
def _objective(x: np.ndarray) -> float:
    # we minimise the negative sum of radii
    return -np.sum(x[2 * N_CIRCLES:])


# ----------------------------------------------------------------------
#  Helper: a feasible hexagonal pattern (25 circles) + one centre circle
# ----------------------------------------------------------------------
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
    radii = np.full(N_CIRCLES, r0)
    radii[-1] = 0.04                     # a little smaller for the middle one

    x0 = np.empty(2 * N_CIRCLES + N_CIRCLES)
    x0[:2 * N_CIRCLES] = centres.ravel()
    x0[2 * N_CIRCLES:] = radii
    return x0


# ----------------------------------------------------------------------
#  Perturb a start vector (uniform jitter)
# ----------------------------------------------------------------------
def _perturbed_guess(base_x: np.ndarray,
                     rng: np.random.Generator,
                     cen_scale: float = 0.025,
                     rad_scale: float = 0.025) -> np.ndarray:
    """Return a copy of base_x with a small uniform random perturbation."""
    x = base_x.copy()
    centres = x[:2 * N_CIRCLES].reshape(-1, 2)
    radii = x[2 * N_CIRCLES:]

    centres += rng.uniform(-cen_scale, cen_scale, size=centres.shape)
    radii += rng.uniform(-rad_scale, rad_scale, size=radii.shape)
    radii = np.maximum(radii, 0.0)

    # clip centres to the unit square (still feasible because radii are tiny)
    centres = np.clip(centres, 0.0, 1.0)

    x[:2 * N_CIRCLES] = centres.ravel()
    x[2 * N_CIRCLES:] = radii
    return x


# ----------------------------------------------------------------------
#  One SLSQP optimisation starting from a given vector
# ----------------------------------------------------------------------
def _solve_one_start(x0: np.ndarray) -> np.ndarray:
    constraints = _make_constraints()
    bounds = [(None, None)] * (2 * N_CIRCLES) + [(0.0, None)] * N_CIRCLES

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
    radii = final[2 * N_CIRCLES:]
    radii = np.where(radii < 0.0, 0.0, radii)
    final[2 * N_CIRCLES:] = radii
    return final


# ----------------------------------------------------------------------
#  Deterministic hill‑climbing on the centre positions
# ----------------------------------------------------------------------
def _local_search(centres_start: np.ndarray,
                  rng: np.random.Generator,
                  max_iters: int = 8000,
                  init_eps: float = 0.025,
                  decay: float = 0.995) -> Tuple[np.ndarray, np.ndarray, float]:
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

        if total > best_total + 1e-12:
            best_c = proposal
            best_r = rad
            best_total = total
            eps = max(eps * 0.9, 1e-5)          # tighten after improvement
        else:
            eps *= decay
            if eps < 1e-6:
                break

    return best_c, best_r, best_total


# ----------------------------------------------------------------------
#  Fine‑grained simulated annealing / long random walk
# ----------------------------------------------------------------------
def _simulated_annealing(centres_start: np.ndarray,
                         rng: np.random.Generator,
                         n_iters: int = 60000,
                         sigma_start: float = 0.03,
                         sigma_end: float = 0.001,
                         temp_start: float = 0.02,
                         temp_end: float = 5e-7) -> Tuple[np.ndarray, np.ndarray, float]:
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
        if delta > 0.0 or rng.random() < np.exp(delta / max(temp, 1e-12)):
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


# ----------------------------------------------------------------------
#  Random initial guess (centres only, radii will be set by the LP)
# ----------------------------------------------------------------------
def _random_centre_guess(rng: np.random.Generator) -> np.ndarray:
    """
    Returns a flat vector containing N_CIRCLES random centre coordinates.
    Radii are initialised to a tiny positive number just to keep the
    SLSQP start feasible (they will be replaced by the LP anyway).
    """
    centres = rng.uniform(0.0, 1.0, size=(N_CIRCLES, 2))
    # initialise radii to a small uniform value – this avoids negative
    # radii warnings in the SLSQP start vector
    radii = np.full(N_CIRCLES, 0.02)

    x0 = np.empty(2 * N_CIRCLES + N_CIRCLES)
    x0[:2 * N_CIRCLES] = centres.ravel()
    x0[2 * N_CIRCLES:] = radii
    return x0


# ----------------------------------------------------------------------
#  Main entry point
# ----------------------------------------------------------------------
def run_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a feasible packing of 26 circles in the unit square that
    maximises the sum of radii.  Returns
        (centres, radii, sum_of_radii)
    where ``centres`` has shape (26,2) and ``radii`` has shape (26,).
    """
    rng = np.random.default_rng(seed=0)   # deterministic RNG

    # ------------------------------------------------------------------
    #  Stage 0 – baseline hexagonal layout (provides a good start)
    # ------------------------------------------------------------------
    base_vec = _initial_guess_hex()
    base_centres = base_vec[:2 * N_CIRCLES].reshape((N_CIRCLES, 2))

    best_c = base_centres.copy()
    best_r = _refine_radii(best_c)
    best_total = float(best_r.sum())

    # ------------------------------------------------------------------
    #  Stage 1 – many short SLSQP runs (exploration)
    # ------------------------------------------------------------------
    n_starts = 250                     # more restarts than the original code
    for k in range(n_starts):
        if k == 0:
            start = base_vec
        elif k <= 30:
            # modest jitter of the original hex layout
            start = _perturbed_guess(base_vec, rng,
                                     cen_scale=0.04,
                                     rad_scale=0.04)
        else:
            # completely random centre configuration
            start = _random_centre_guess(rng)

        sol = _solve_one_start(start)

        centres = sol[:2 * N_CIRCLES].reshape((N_CIRCLES, 2))
        radii = _refine_radii(centres)
        total = float(radii.sum())

        if total > best_total + 1e-12:
            best_c = centres
            best_r = radii
            best_total = total

    # ------------------------------------------------------------------
    #  Stage 2 – deterministic hill‑climbing on the centre positions
    # ------------------------------------------------------------------
    best_c, best_r, best_total = _local_search(
        best_c,
        rng,
        max_iters=12000,
        init_eps=0.030,
        decay=0.996
    )

    # ------------------------------------------------------------------
    #  Stage 3 – fine‑grained simulated annealing / long random walk
    # ------------------------------------------------------------------
    best_c, best_r, best_total = _simulated_annealing(
        best_c,
        rng,
        n_iters=80000,            # extended annealing phase
        sigma_start=0.04,
        sigma_end=0.001,
        temp_start=0.025,
        temp_end=5e-7
    )

    # ------------------------------------------------------------------
    #  Stage 4 – final polish with SLSQP (uses the best found layout)
    # ------------------------------------------------------------------
    final_vec = np.empty(2 * N_CIRCLES + N_CIRCLES)
    final_vec[:2 * N_CIRCLES] = best_c.ravel()
    final_vec[2 * N_CIRCLES:] = best_r
    final_vec = _solve_one_start(final_vec)

    final_centres = final_vec[:2 * N_CIRCLES].reshape((N_CIRCLES, 2))
    final_radii = _refine_radii(final_centres)
    final_total = float(final_radii.sum())

    # ------------------------------------------------------------------
    #  Return the best solution discovered
    # ------------------------------------------------------------------
    return final_centres, final_radii, final_total