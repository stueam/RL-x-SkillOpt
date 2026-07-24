You are an expert in extremal combinatorics.

Your task is to write a Python function `construct(n, seed)` that builds a large cap set in F_3^n (the vector space over GF(3) with dimension n).

{skill_section}

## Task Format
You will receive the dimension n and a random seed.

## Required Interface
```python
def construct(n: int, seed: int) -> np.ndarray:
    """Returns a NumPy array of shape (m, n), elements in {{0,1,2}}, representing a cap set of size m."""
```

## Cap set definition
A **cap set** in F_3^n is a subset with no three-term arithmetic progression:
- Three distinct vectors x, y, z in F_3^n form a **bad triple** if x + y + z ≡ 0 (mod 3) for ALL coordinates
- Equivalently: x_i + y_i + z_i ≡ 0 (mod 3) for every coordinate i

## Validation (your output will be checked against these rules)
1. Shape must be (m, n) with m >= 1
2. All elements must be 0, 1, or 2
3. No duplicate rows
4. No three distinct elements sum to 0 mod 3 (the cap property)
   - Efficient check: for each pair (a,b), compute z = (-a-b) mod 3. If z is in the set and z != a and z != b, the set is invalid.

## Randomness
Use `np.random.RandomState(seed)` to shuffle candidate order. Different seeds should produce different cap sets of potentially different sizes.

## Known optimal sizes
- n=4: 20, n=5: 45, n=6: 112

## Common Mistakes
- Missing the cap property check entirely
- Not using the seed (deterministic results across calls)
- Returning wrong shape or value range

## Output Format
Think step by step, then output your code between ```python and ```.
