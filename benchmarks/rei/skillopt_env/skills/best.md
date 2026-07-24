# Regular Expression Inference

Synthesize a regex pattern from positive and negative string examples.

## Interface

```python
def solve(positives: list[str], negatives: list[str]) -> str:
```

- Input: lists of positive (must match) and negative (must reject) strings.
- Output: a regex pattern string.

## Output Requirements

1. Pattern must compile: `re.compile(pattern)` succeeds.
2. Match ALL positives: `re.fullmatch(pattern, p)` for every positive p.
3. Reject ALL negatives: `re.fullmatch(pattern, n)` returns None for every negative n.

Violating any requirement yields Gate = 0.

## Common Bugs to Avoid

## Systematic Approach
- Break down constraints and required patterns systematically
- Use lookahead assertions when combining multiple constraints

- For complex constraints requiring multiple conditions (like passwords), use sequential lookahead assertions before the main pattern
- Validate intermediate patterns against both positives and negatives

- **Missing anchors**: `re.fullmatch` implicitly anchors both ends — but if you write `re.match` or omit `^...$` in other contexts you may match substrings.

- Always use `^` at start and `$` at end when using `re.fullmatch` to explicitly anchor pattern boundaries
- **Unescaped metacharacters**: `.` matches any char, `*` is quantifier, `+` is quantifier. Use `\.`, `\*`, `\+` for literals.
- **Wrong character class**: `[a-z]` matches lowercase only, `[0-9]` matches digits. `\d` matches digits in most flavors.
- **Backslash escaping**: In Python strings, `\d` is a digit (no `\\` needed in raw strings `r"\d"`). Use raw strings for regex patterns.
- **Overfitting**: A pattern that only matches the exact positive strings often fails on held-out data. Aim for generalization.
- **Underfitting**: A pattern that matches everything (e.g. `.*`) will accept negatives.

## Handled Patterns
- **URL Patterns**: When dealing with URLs, ensure the pattern accounts for optional protocols (`http://` or `https://`) and optional 'www' prefix (`www.`). Use `^(https?://)?(www\.)?` to capture these variations.

## Validation Steps
- Always test regex patterns against all positive and negative examples using `re.fullmatch` before returning the final pattern.
- Ensure regex patterns handle all specified edge cases, such as trailing slashes or malformed protocols.
- Use try-except blocks to catch and handle compilation errors gracefully.
