"""Runtime shape-checked array typing via jaxtyping and typeguard."""

from typing import Final

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
    "typed",
)

typed: Final = jaxtyped(typechecker=typeguard.typechecked)
