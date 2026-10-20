"""Print or generate the canonical authored-config JSON Schema and field metadata."""

import argparse
import json
from pathlib import Path

from config.json_schema import generate_schema


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "schemas" / "config-v9.schema.json"


def render_schema():
    return json.dumps(generate_schema(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write schema to this path instead of stdout")
    parser.add_argument("--check", action="store_true", help="check the generated file without writing")
    parser.add_argument("--field", metavar="PATH", help="print a field metadata report for editor/explain use")
    args = parser.parse_args(argv)
    if args.field:
        if args.output or args.check:
            parser.error("--field cannot be combined with --output or --check")
        from config.field_registry import describe_field

        try:
            print(json.dumps(describe_field(args.field), indent=2, ensure_ascii=False, allow_nan=False))
        except KeyError as exc:
            parser.error(str(exc))
        return 0
    rendered = render_schema()
    output = args.output or DEFAULT_OUTPUT
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            print(f"stale generated configuration schema: {output}")
            return 1
        print(f"generated configuration schema is current: {output}")
    elif args.output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
