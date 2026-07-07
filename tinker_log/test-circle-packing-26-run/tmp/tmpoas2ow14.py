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
    for i in range(n_circles):
        cons.append({'type': 'ineq', 'fun': InsideConstraint(i)})
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
        options={'ftol': 1e-12, 'maxiter': 8000, 'disp': False}
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
# Simple greedy hill‑climbing that moves only the centres.
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

    for _ in range(max_iters):
        proposal = best_c + rng.uniform(-eps, eps, size=best_c.shape)
        proposal = np.clip(proposal, 0.0, 1.0)

        rad = _refine_radii(proposal)
        total = float(rad.sum())

        if total > best_total + 1e-12:
            best_c = proposal
            best_r = rad
            best_total = total
            eps = max(eps * 0.9, 1e-5)
        else:
            eps *= decay
            if eps < 1e-6:
                break

    return best_c, best_r, best_total


# ----------------------------------------------------------------------
# Simulated annealing search in centre space (LP evaluates each state)
# ----------------------------------------------------------------------
def _annealing_search(centres_start: np.ndarray,
                      rng: np.random.Generator,
                      max_iters: int = 50000,
                      init_step: float = 0.05,
                      step_decay: float = 0.9999,
                      temp_start: float = 0.01,
                      temp_decay: float = 0.9999,
                      mutation_prob: float = 0.001,
                      mutation_range: float = 0.3) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulated annealing over centre positions.
    Each state is evaluated by the LP that gives the optimal radii.
    """
    best_c = centres_start.copy()
    best_r = _refine_radii(best_c)
    best_total = float(best_r.sum())

    cur_c = best_c.copy()
    cur_total = best_total

    step = init_step
    temp = temp_start

    n = best_c.shape[0]

    for _ in range(max_iters):
        # propose a move
        proposal = cur_c + rng.uniform(-step, step, size=cur_c.shape)

        # occasional large mutation of a single circle
        if rng.random() < mutation_prob:
            idx = rng.integers(0, n)
            proposal[idx] = rng.uniform(0.0, 1.0, size=2)

        # occasionally swap two circles (helps escape symmetric traps)
        if rng.random() < mutation_prob * 0.5:
            i, j = rng.choice(n, size=2, replace=False)
            proposal[[i, j]] = proposal[[j, i]]

        proposal = np.clip(proposal, 0.0, 1.0)

        rad = _refine_radii(proposal)
        total = float(rad.sum())
        delta = total - cur_total

        # Acceptance test
        if delta > 0.0 or rng.random() < np.exp(delta / max(temp, 1e-12)):
            cur_c = proposal
            cur_total = total
            if total > best_total:
                best_c = proposal
                best_r = rad
                best_total = total

        # Update schedule
        step = max(step * step_decay, 1e-6)
        temp = max(temp * temp_decay, 1e-12)

    return best_c, best_r, best_total


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
    # Stage 1 – modest number of SLSQP restarts (quick global exploration)
    # ------------------------------------------------------------------
    n_starts = 30
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
                break

    # ------------------------------------------------------------------
    # Stage 2 – greedy hill‑climbing (already in the original code)
    # ------------------------------------------------------------------
    if best_total < target:
        best_centres, best_radii, best_total = _local_search(
            best_centres,
            rng,
            max_iters=5000,
            init_eps=0.025,
            decay=0.998
        )

    # ------------------------------------------------------------------
    # Stage 3 – simulated annealing on centre positions (global refinement)
    # ------------------------------------------------------------------
    if best_total < target:
        best_centres, best_radii, best_total = _annealing_search(
            best_centres,
            rng,
            max_iters=50000,
            init_step=0.05,
            step_decay=0.9999,
            temp_start=0.01,
            temp_decay=0.9999,
            mutation_prob=0.001,
            mutation_range=0.3
        )

    # ------------------------------------------------------------------
    # Stage 4 – final polishing with SLSQP (centres + radii together)
    # ------------------------------------------------------------------
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