"""Runtime shape-checked array typing via jaxtyping and typeguard."""

from collections.abc import Generator
import contextlib
from typing import Final

import jaxtyping
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    Shaped,
    TypeCheckError,
    jaxtyped,
)
import typeguard

__all__ = (
    "Array",
    "Bool",
    "Float",
    "Int",
    "Shaped",
    "TypeCheckError",
    "disable_typechecking",
    "typed",
)

typed: Final = jaxtyped(typechecker=typeguard.typechecked)


@contextlib.contextmanager
def disable_typechecking() -> Generator[None]:
    """Temporarily disables runtime jaxtyping checks for shape-only tree traversal."""
    previous = jaxtyping.config.jaxtyping_disable
    jaxtyping.config.update("jaxtyping_disable", value=True)
    try:
        yield
    finally:
        jaxtyping.config.update("jaxtyping_disable", value=previous)
