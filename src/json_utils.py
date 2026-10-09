"""Readable JSON formatting without application or exchange dependencies."""

import json
from typing import Any, Callable


class _ObjectPairs(list):
    """Keep object members distinct even when encoded keys collide."""


class _NumberToken(str):
    """A validated JSON number whose original spelling must survive formatting."""


def reformat_json_text(
    text: str,
    *,
    indent: int = 4,
    max_inline: int = 72,
    sort_keys: bool = False,
) -> str:
    """Format strict JSON without rounding numbers or discarding duplicate members."""

    def reject_constant(value):
        raise ValueError(f"{value} is not a valid JSON number")

    data = json.loads(
        text,
        object_pairs_hook=_ObjectPairs,
        parse_int=_NumberToken,
        parse_float=_NumberToken,
        parse_constant=reject_constant,
    )
    if sort_keys:

        def sort(node):
            if isinstance(node, _ObjectPairs):
                return _ObjectPairs(
                    (key, sort(value))
                    for key, value in sorted(node, key=lambda pair: pair[0])
                )
            if isinstance(node, list):
                return [sort(value) for value in node]
            return node

        data = sort(data)
    return _render_streamlined(data, indent=indent, max_inline=max_inline)


def dump_json_streamlined(
    data: Any,
    fp,
    *,
    indent: int | None = 4,
    max_inline: int = 60,
    separators: tuple[str, str] = (", ", ": "),
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Write :func:`json_dumps_streamlined` output to a file-like object."""
    fp.write(
        json_dumps_streamlined(
            data,
            indent=indent,
            max_inline=max_inline,
            separators=separators,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            allow_nan=allow_nan,
            default=default,
        )
    )


def json_dumps_streamlined(
    data: Any,
    *,
    indent: int | None = 4,
    max_inline: int = 60,
    separators: tuple[str, str] = (", ", ": "),
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
    default: Callable[[Any], Any] | None = None,
) -> str:
    """Keep short JSON containers inline and indent larger containers.

    ``max_inline`` counts the serialized container, including its delimiters.
    Serialization options retain the semantics of ``json.dumps``; ``indent=None``
    returns its ordinary single-line output. No trailing newline is added.
    """
    serialized = json.dumps(
        data,
        separators=separators,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
        default=default,
    )
    if indent is None or len(serialized) <= max_inline:
        return serialized

    # Normalize once through the standard encoder: custom values, tuple values,
    # dictionary keys and validation must behave the same for every block size.
    normalized = json.loads(serialized, object_pairs_hook=_ObjectPairs)
    return _render_streamlined(
        normalized,
        indent=indent,
        max_inline=max_inline,
        separators=separators,
        ensure_ascii=ensure_ascii,
    )


def _render_streamlined(
    normalized, *, indent, max_inline, separators=(", ", ": "), ensure_ascii=True
):

    def compact(value: Any) -> str:
        if isinstance(value, _NumberToken):
            return str(value)
        if isinstance(value, _ObjectPairs):
            entries = [
                f"{json.dumps(key, ensure_ascii=ensure_ascii)}{separators[1]}{compact(val)}"
                for key, val in value
            ]
            return "{" + separators[0].join(entries) + "}"
        if isinstance(value, list):
            return "[" + separators[0].join(compact(item) for item in value) + "]"
        return json.dumps(value, ensure_ascii=ensure_ascii)

    def render(value: Any, level: int) -> str:
        inline = compact(value)
        if len(inline) <= max_inline or not isinstance(value, list) or not value:
            return inline

        indent_str = " " * (indent * level)
        child_indent = " " * (indent * (level + 1))
        if isinstance(value, _ObjectPairs):
            opening, closing = "{", "}"
            entries = [
                f"{json.dumps(key, ensure_ascii=ensure_ascii)}{separators[1]}{render(val, level + 1)}"
                for key, val in value
            ]
        else:
            opening, closing = "[", "]"
            entries = [render(item, level + 1) for item in value]
        separator = separators[0].rstrip() + "\n" + child_indent
        return (
            opening
            + "\n"
            + child_indent
            + separator.join(entries)
            + "\n"
            + indent_str
            + closing
        )

    return render(normalized, 0)
