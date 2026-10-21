"""Explicit field metadata; independent of loading and trading runtimes.

Descriptors document serialization types. They do not replace the owning
validators, choose defaults at runtime, or grant permission to override a field.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable


class FieldType(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"


class _MissingDefault(Enum):
    VALUE = "no default"


NO_DEFAULT = _MissingDefault.VALUE


class CoinOverridePolicy(str, Enum):
    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"
    COIN_HSL_ONLY = "coin HSL scope only"


@dataclass(frozen=True)
class FieldDomain:
    """Declarative domain metadata; owning validators still enforce values."""

    minimum: int | float | None = None
    maximum: int | float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    finite: bool | None = None
    choices: tuple[Any, ...] = ()
    # A pattern describes accepted authored strings; choices then provide
    # canonical examples rather than an exhaustive raw-input enumeration.
    string_pattern: str | None = None


@dataclass(frozen=True)
class FieldSpec:
    path: tuple[str, ...]
    types: tuple[FieldType, ...]
    nullable: bool = False
    default: Any = NO_DEFAULT
    source: str = "python"
    unit: str | None = None
    description: str = ""
    optimize_key: str | None = None
    optimizable: bool | None = None
    optimize_when: str | None = None
    scope: str | None = None
    consumers: tuple[str, ...] = ()
    domain: FieldDomain | None = None
    coin_override: CoinOverridePolicy | None = None
    order: int = 0
    recommended_bounds: tuple[int | float, ...] | None = None
    denominator: str | None = None

    def __post_init__(self):
        if not self.path or any(not isinstance(part, str) or not part for part in self.path):
            raise ValueError("field paths must contain non-empty string segments")
        if not self.types or any(not isinstance(kind, FieldType) for kind in self.types):
            raise TypeError("field types must be explicitly declared FieldType values")
        if len(set(self.types)) != len(self.types):
            raise ValueError(f"duplicate types for {'.'.join(self.path)}")
        if FieldType.FLOAT in self.types and FieldType.INTEGER in self.types:
            raise ValueError("a field must choose one numeric serialization type")

    @property
    def numeric_type(self) -> type | None:
        if FieldType.FLOAT in self.types:
            return float
        if FieldType.INTEGER in self.types:
            return int
        return None

    @property
    def type_label(self) -> str:
        labels = [kind.value for kind in self.types]
        if self.nullable:
            labels.append("null")
        return " | ".join(labels)


class FieldRegistry:
    """Index exact paths and declared one-segment wildcard paths.

    Wildcards describe record members and dynamic map keys, never alternate
    public spellings. A literal field always takes precedence over a wildcard.
    """

    def __init__(self, fields: Iterable[FieldSpec]):
        exact = {}
        patterns = []
        for field in fields:
            if field.path in exact:
                raise ValueError(f"duplicate field path {'.'.join(field.path)}")
            exact[field.path] = field
            if "*" in field.path:
                patterns.append(field)
        self._fields = MappingProxyType(exact)
        self._patterns = tuple(patterns)

    def __iter__(self):
        return iter(self._fields.values())

    def find(self, path: tuple[str | int, ...]) -> FieldSpec | None:
        exact = self._fields.get(path)
        if exact is not None:
            return exact
        matches = [
            field for field in self._patterns
            if len(field.path) == len(path)
            and all(expected == "*" or expected == actual
                    for expected, actual in zip(field.path, path))
        ]
        if not matches:
            return None
        specificity = max(sum(part != "*" for part in field.path) for field in matches)
        matches = [field for field in matches
                   if sum(part != "*" for part in field.path) == specificity]
        if len(matches) != 1:
            raise ValueError(f"ambiguous field descriptors for {path!r}")
        return matches[0]
