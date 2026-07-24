import numpy as np
from itertools import product, combinations

_OPTIMAL = {1: 2, 2: 4, 3: 9, 4: 20, 5: 45, 6: 112, 7: 236, 8: 496}


def is_cap_set(A: np.ndarray) -> bool:
    """Check if A (m, n) is a valid cap set."""
    m = A.shape[0]
    if m < 3:
        return True
    Aset = set(map(tuple, A))
    for i in range(m):
        ai = A[i]
        for j in range(i + 1, m):
            z = tuple(np.mod(-(ai + A[j]), 3))
            if z in Aset and z != tuple(ai) and z != tuple(A[j]):
                return False
    return True


def evaluate_construct(construct_fn, n: int = 4) -> dict:
    """Gate × Quality evaluation for a construct(n) function."""
    try:
        A = construct_fn(n)
        A = np.asarray(A, dtype=int)
        if A.ndim != 2 or A.shape[1] != n or A.shape[0] < 1:
            return {"gate": 0, "quality": 0, "error": "shape"}
        if not np.all((A >= 0) & (A <= 2)):
            return {"gate": 0, "quality": 0, "error": "values"}
        if len(set(map(tuple, A))) != A.shape[0]:
            return {"gate": 0, "quality": 0, "error": "dup"}
        if not is_cap_set(A):
            return {"gate": 0, "quality": 0, "error": "not_cap"}
        return {"gate": 1, "quality": int(A.shape[0]), "error": ""}
    except Exception as e:
        return {"gate": 0, "quality": 0, "error": str(e)[:50]}
