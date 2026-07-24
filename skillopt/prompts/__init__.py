"""Prompt loading utilities for ReflACT.

Prompts are stored as ``.md`` files and loaded at runtime:

- **Generic** prompts live in ``skillopt/prompts/*.md``
- **Env-specific** prompts live in ``skillopt/envs/<env>/prompts/*.md``

``load_prompt(name, env)`` tries the env-specific path first, then falls
back to the generic default.
"""
from __future__ import annotations

import os

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REFLACT_DIR = os.path.dirname(_PROMPTS_DIR)

_cache: dict[str, str] = {}


def _read_file(path: str) -> str | None:
    if path in _cache:
        return _cache[path]
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        content = f.read()
    _cache[path] = content
    return content


_ROOT = os.path.dirname(_REFLACT_DIR)


def load_prompt(name: str, env: str | None = None) -> str:
    """Load a prompt by name with env-specific override and generic fallback.

    Lookup order:
      1. ``skillopt/envs/{env}/prompts/{name}.md``  (if *env* given)
      2. ``benchmarks/{env}/skillopt_env/prompts/{name}.md``  (if *env* given)
      3. ``skillopt/prompts/{name}.md``              (generic default)

    Raises ``FileNotFoundError`` if neither path exists.
    """
    searched = []
    if env is not None:
        for base in (
            os.path.join(_REFLACT_DIR, "envs", env, "prompts", f"{name}.md"),
            os.path.join(_ROOT, "benchmarks", env, "skillopt_env", "prompts", f"{name}.md"),
        ):
            content = _read_file(base)
            searched.append(os.path.relpath(base, _ROOT))
            if content is not None:
                return content

    generic_path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    content = _read_file(generic_path)
    if content is not None:
        return content
    searched.append(f"skillopt/prompts/{name}.md")

    raise FileNotFoundError(
        f"Prompt '{name}' not found. Searched: {', '.join(searched)}"
    )


def clear_cache() -> None:
    """Clear the prompt file cache (useful for testing)."""
    _cache.clear()
