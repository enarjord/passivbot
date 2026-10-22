"""Explicitly replace a config's optimizer search space with recommended ranges.

This is an offline authoring tool. Loading a sparse config never applies this
preset automatically. The output is a new file; the input is never overwritten.
"""

import argparse
from copy import deepcopy
from pathlib import Path


def _anchor_file_references(config, *, base_config_path):
    """Retain file authority without depending on the exported config's location."""
    from config.overrides import _resolve_override_config_path
    from config.universe import normalize_universe_source_bindings, resolve_universe_source_path

    base = base_config_path or config.get("live", {}).get("base_config_path", "")

    # Initial selection may use the established repo/CWD/legacy fallbacks.
    # Once selected, anchoring retains that binding rather than inventing one.
    normalize_universe_source_bindings(config, base_config_path=base)
    normalize_universe_source_bindings(config, anchor_all=True, base_config_path=base)

    def visit(node, prefix=()):
        if isinstance(node, list):
            for item in node:
                visit(item, prefix)
        elif isinstance(node, dict):
            for key, value in node.items():
                parts = (*prefix, *key.split("."))
                if parts[-1] == "override_config_path" and "coin_overrides" in parts:
                    context = {
                        "live": {"base_config_path": base},
                        "coin_overrides": {"source": {"override_config_path": value}},
                    }
                    node[key] = _resolve_override_config_path(context, "source")
                elif (len(parts) >= 4 and parts[-4] == "bot"
                      and parts[-3] in ("global", "long", "short") and parts[-2] == "universe"
                      and parts[-1] in ("approved_coins", "ignored_coins")):
                    candidate = value[0] if isinstance(value, list) and len(value) == 1 else value
                    if isinstance(candidate, str):
                        resolved = resolve_universe_source_path(candidate, base_config_path=base, allow_all=parts[-1] == "approved_coins")
                        if resolved is not None:
                            node[key] = [str(resolved)] if isinstance(value, list) else str(resolved)
                else:
                    visit(value, parts)

    visit(config)


def apply_recommended_bounds(source, *, base_config_path=""):
    from config import prepare_config
    from config.authored import validate_authored_config
    from config.hsl import generated_template
    from config.migrations.v9 import require_v9_runtime_config
    from config.schema import get_template_config
    from config_utils import strip_config_metadata

    require_v9_runtime_config(source)
    template = get_template_config()
    validate_authored_config(source, template)
    result = strip_config_metadata(source)
    mode = result.get("bot", {}).get("portfolio", {}).get("hsl", {}).get("scope", "coin")
    profile = generated_template(template, mode)
    result.setdefault("optimize", {})["bounds"] = deepcopy(profile["optimize"]["bounds"])
    # Unilateralness is optional in the starting profile. A positive fixed
    # coordinate cannot coexist with varying normalized sibling weights.
    for side in ("long", "short"):
        weights = result.get("bot", {}).get(side, {}).get("forager", {}).get("score_weights", {})
        if weights.get("unilateralness", 0.0) > 0:
            result["optimize"]["bounds"][side]["forager"].setdefault("score_weights", {})["unilateralness"] = [0.0, 1.0, 0.01]
    _anchor_file_references(result, base_config_path=base_config_path)
    result = prepare_config(result, base_config_path=base_config_path, verbose=False)
    result = strip_config_metadata(result)
    result["live"].pop("base_config_path", None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Sparse authored v9 bounds freeze omitted independent genes. get_template_config() "
            "and the no-config optimizer command provide a complete starting profile "
            "with explicit recommended ranges."
        ),
    )
    parser.add_argument("input", type=Path, help="Authored v9 configuration")
    parser.add_argument("output", type=Path, help="New config containing explicit bounds")
    parser.add_argument("--preset", choices=("recommended",), default="recommended")
    args = parser.parse_args(argv)
    from config import load_input_config
    from json_utils import json_dumps_streamlined

    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    source, base, _raw = load_input_config(str(args.input), log_info=False)
    result = apply_recommended_bounds(source, base_config_path=base)
    with args.output.open("x") as output:
        output.write(json_dumps_streamlined(result) + "\n")
    print(f"Wrote explicit recommended optimizer bounds to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
