from __future__ import annotations

import os


def env_str(name: str, default: str) -> str:
    """Read an environment variable as a string (or return default)."""
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    """Read an environment variable as an int (or return default)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an int, got {raw!r}") from e


def env_float(name: str, default: float) -> float:
    """Read an environment variable as a float (or return default)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a float, got {raw!r}") from e


def env_choice(name: str, default: str, choices: set[str]) -> str:
    """Read an env var constrained to a fixed set of choices."""
    val = os.getenv(name, default)
    if val not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {val!r}")
    return val
