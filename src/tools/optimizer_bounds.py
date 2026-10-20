"""Explicitly replace a config's optimizer search space with recommended ranges.

This is an offline authoring tool. Loading a sparse config never applies this
preset automatically. The output is a new file; the input is never overwritten.
"""

import argparse
from copy import deepcopy
from pathlib import Path


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
    mode = result.get("bot", {}).get("global", {}).get("hsl", {}).get("scope", "coin")
    profile = generated_template(template, mode)
    result.setdefault("optimize", {})["bounds"] = deepcopy(profile["optimize"]["bounds"])
    # Unilateralness is optional in the starting profile. A positive fixed
    # coordinate cannot coexist with varying normalized sibling weights.
    for side in ("long", "short"):
        weights = result.get("bot", {}).get(side, {}).get("forager", {}).get("score_weights", {})
        if weights.get("unilateralness", 0.0) > 0:
            result["optimize"]["bounds"][side]["forager"].setdefault("score_weights", {})["unilateralness"] = [0.0, 1.0, 0.01]
    result = prepare_config(result, base_config_path=base_config_path, verbose=False)
    result = strip_config_metadata(result)
    result["live"].pop("base_config_path", None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Sparse authored v9 bounds freeze omitted dimensions. get_template_config() "
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
