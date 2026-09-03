"""Internal helpers for process-local runtime configuration."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import os
from typing import Any, Iterator


EnvironmentValue = str | os.PathLike[str] | None
Environment = Mapping[str, EnvironmentValue]


def normalize_environment(env: Environment | None) -> dict[str, str | None]:
    """Return a string-valued environment overlay.

    ``None`` values mean that an inherited variable should be removed.  The
    normalized mapping is safe to store in ASE's ``Parameters`` object and to
    pickle for a worker process.
    """

    if env is None:
        return {}
    if not isinstance(env, Mapping):
        raise TypeError("env must be a mapping of variable names to values")
    normalized: dict[str, str | None] = {}
    for key, value in env.items():
        name = os.fsdecode(key) if isinstance(key, bytes) else str(key)
        if value is None:
            normalized[name] = None
        elif isinstance(value, bytes):
            normalized[name] = os.fsdecode(value)
        elif isinstance(value, os.PathLike):
            normalized[name] = os.fspath(value)
        else:
            normalized[name] = str(value)
    return normalized


def merged_environment(env: Environment | None) -> dict[str, str]:
    """Merge an environment overlay into a copy of the parent environment."""

    result = dict(os.environ)
    for key, value in normalize_environment(env).items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


@contextmanager
def temporary_environment(env: Environment | None) -> Iterator[None]:
    """Temporarily apply an environment overlay to the current process."""

    normalized = normalize_environment(env)
    missing = object()
    previous: dict[str, Any] = {
        key: os.environ.get(key, missing) for key in normalized
    }
    try:
        for key, value in normalized.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve_alias(values: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    """Resolve equal optional aliases and reject conflicting values."""

    provided = [
        (name, values[name])
        for name in names
        if name in values and values[name] is not None
    ]
    if not provided:
        return None
    first_name, first_value = provided[0]
    for name, value in provided[1:]:
        if value != first_value:
            raise ValueError(
                f"conflicting aliases for {names[0]}: "
                f"{first_name}={first_value!r} and {name}={value!r}"
            )
    return first_value


def normalize_optional_int(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> int | None:
    """Validate and normalize a Python integer-valued optional setting."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or None")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer or None") from exc
    if normalized != value:
        raise ValueError(f"{name} must be an integer or None")
    if normalized < minimum:
        comparison = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {comparison}")
    return normalized
