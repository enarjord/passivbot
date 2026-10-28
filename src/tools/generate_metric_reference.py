"""Generate the analysis quantity reference from immutable metric definitions."""

import argparse
from pathlib import Path

from config.metric_definitions import METRIC_DEFINITIONS


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "docs/ai/generated/metric_fields.md"


def _cell(value):
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def render_reference():
    lines = [
        "# Analysis Quantity Reference",
        "",
        "Generated from `src/config/metric_definitions.py`; do not edit by hand.",
        "The Rust-owned core catalog supplies the 26 exact selectors for 25 quantities.",
        "This describes quantities, not objective direction, selector aliases, reductions or backend support.",
        "Core definition version 1 owns values, clocks and status semantics; historical definitions require re-evaluation.",
        "",
        "| Metric | Unit / denominator | Basis | Sampling / intrinsic aggregation | Scope / weighting | Sign / domain / notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, definition in sorted(METRIC_DEFINITIONS.items()):
        unit = definition.unit + (f" / {definition.denominator}" if definition.denominator else "")
        scope = ", ".join(item for item in (definition.curve, definition.position_side, definition.weighting) if item)
        notes = "; ".join(item for item in (definition.sign, definition.domain, definition.notes) if item)
        cells = (f"`{name}`", unit, definition.basis,
                 f"{definition.sampling}; {definition.aggregation}", scope, notes)
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    content = render_reference()
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != content:
            print("stale generated analysis quantity reference")
            return 1
        print("generated analysis quantity reference is current")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
