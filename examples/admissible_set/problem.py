import itertools
import numpy as np

_TRIPLES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1), (2, 2, 2)]
_INT_TO_WEIGHT = [0, 1, 1, 2, 2, 3, 3]
_BAD_TRIPLES = {
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5), (0, 0, 6),
    (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5), (0, 6, 6),
    (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3), (1, 2, 4),
    (1, 3, 3), (1, 4, 4), (1, 5, 5), (1, 6, 6),
    (2, 2, 2), (2, 3, 3), (2, 4, 4), (2, 5, 5), (2, 6, 6),
    (3, 3, 3), (3, 3, 4), (3, 4, 4), (3, 4, 5), (3, 4, 6), (3, 5, 5), (3, 6, 6),
    (4, 4, 4), (4, 5, 5), (4, 6, 6),
    (5, 5, 5), (5, 5, 6), (5, 6, 6),
    (6, 6, 6),
}
_OPTIMAL = {(12, 7): 792, (15, 10): 3003, (21, 15): 43596, (24, 17): 237984}


def generate_candidates(dimension, weight):
    num_groups = dimension // 3
    return [
        np.array(c, dtype=np.int32)
        for c in itertools.product(range(7), repeat=num_groups)
        if sum(_INT_TO_WEIGHT[x] for x in c) == weight
    ]


def _expand_child(child, num_groups):
    result = []
    for i in range(num_groups):
        x, y, z = _TRIPLES[child[i]]
        result.append((x, y, z))
        if not (x == y == z):
            result.append((z, x, y))
            result.append((y, z, x))
    expanded = sum(result, ())
    return expanded


def _get_surviving_indices(extant, new_el, candidates):
    surviving = []
    for idx, child in enumerate(candidates):
        if all(_INT_TO_WEIGHT[x] <= _INT_TO_WEIGHT[y] for x, y in zip(new_el, child)):
            continue
        if all(_INT_TO_WEIGHT[x] >= _INT_TO_WEIGHT[y] for x, y in zip(new_el, child)):
            continue
        invalid = False
        for ext in extant:
            if all(tuple(sorted((int(x), int(y), int(z)))) in _BAD_TRIPLES
                   for x, y, z in zip(ext, new_el, child)):
                invalid = True
                break
        if not invalid:
            surviving.append(idx)
    return surviving


def _expand_set(pre_admissible_set, num_groups):
    result = []
    for row in pre_admissible_set:
        rotations = [[] for _ in range(num_groups)]
        for i in range(num_groups):
            x, y, z = _TRIPLES[row[i]]
            rotations[i].append((x, y, z))
            if not (x == y == z):
                rotations[i].append((z, x, y))
                rotations[i].append((y, z, x))
        for combo in itertools.product(*rotations):
            result.append(sum(combo, ()))
    return result


def evaluate_priority(priority_func, dimension, weight):
    num_groups = dimension // 3
    optimal = _OPTIMAL.get((dimension, weight))
    candidates = generate_candidates(dimension, weight)

    expanded_els = [_expand_child(el, num_groups) for el in candidates]
    scores = np.array([
        priority_func(el, dimension, weight) for el in expanded_els
    ], dtype=float)

    pre_admissible = np.empty((0, num_groups), dtype=np.int32)

    while candidates:
        best = int(np.argmax(scores))
        chosen = candidates[best]
        surviving = _get_surviving_indices(pre_admissible, chosen, candidates)
        candidates = [candidates[i] for i in surviving]
        scores = scores[surviving]
        pre_admissible = np.concatenate([pre_admissible, chosen[None]], axis=0)

    admissible_set = _expand_set(pre_admissible, num_groups)
    achieved = len(admissible_set)
    gap = (optimal - achieved) if optimal else 0
    return achieved, gap
