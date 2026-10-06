#!/usr/bin/env python3
"""Offline config cleanup, canonical exports and lossless JSON formatting."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from json_utils import json_dumps_streamlined, reformat_json_text

MODES = ("full", "live", "backtest", "optimize", "format")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"duplicate object key {key!r}; use --mode format to preserve it"
            )
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"{value} is not a valid JSON number")


def cleanup_config(
    source: dict, *, mode: str = "full", base_config_path: str = ""
) -> dict:
    """Reuse canonical normalization without resolving dates or coin-list sources.

    Exporting is not runtime compilation or strategy migration. In particular,
    never apply optimizer fixed runtime overrides to the authored bot values.
    """
    from config.load import prepare_config
    from config.hydrate import (
        normalize_backtest_coin_sources,
        normalize_optimizer_settings,
    )
    from config.project import project_config
    from config_utils import sanitize_prepared_config_for_dump, strip_config_metadata

    if mode not in MODES[:-1]:
        raise ValueError(f"unsupported config cleanup mode: {mode}")
    if not isinstance(source, dict):
        raise TypeError("config root must be an object")
    payload = source
    if "bot" not in payload and isinstance(payload.get("config"), dict):
        payload = payload["config"]
    if not isinstance(payload.get("bot"), dict) or not isinstance(
        payload.get("live"), dict
    ):
        raise ValueError(
            "expected a V8 config with bot and live objects (possibly inside config)"
        )
    # Ignore result envelopes and validate only the sections relevant to this
    # export. Missing pipeline sections are hydrated by the shared loader.
    target = "canonical" if mode == "full" else mode
    payload = strip_config_metadata(project_config(payload, target, record_step=False))
    for section, value in payload.items():
        if section != "config_version" and not isinstance(value, dict):
            raise TypeError(f"config.{section} must be an object")
    prepared = prepare_config(
        deepcopy(payload),
        base_config_path=base_config_path,
        live_only=True,
        verbose=False,
        log_config_transforms=False,
    )
    if mode != "live":
        prepared["backtest"]["coin_sources"] = normalize_backtest_coin_sources(
            prepared["backtest"]["coin_sources"]
        )
    if mode in ("full", "optimize"):
        raw_optimize = payload.get("optimize", {})
        normalize_optimizer_settings(
            prepared,
            verbose=False,
            raw_optimize_limits=raw_optimize.get("limits"),
            raw_optimize_limits_present="limits" in raw_optimize,
        )
    cleaned = sanitize_prepared_config_for_dump(prepared)
    # These schema leaves accept mappings, lists, sentinel strings and file
    # references. Template-shaped cleaning alone would replace a string/list
    # with the default mapping, even though live normalization preserves it.
    for key in ("approved_coins", "ignored_coins"):
        cleaned["live"][key] = deepcopy(prepared["live"][key])
    # null means inherit the live setting, rather than the schema's false.
    cleaned["backtest"]["filter_by_min_effective_cost"] = prepared["backtest"][
        "filter_by_min_effective_cost"
    ]
    return project_config(cleaned, target, record_step=False)


def render_source(
    text: str,
    *,
    source: Path,
    mode: str,
    input_format: str = "auto",
    indent: int = 4,
    max_inline: int = 72,
    sort_keys: bool = False,
) -> str:
    is_hjson = input_format == "hjson" or (
        input_format == "auto" and source.suffix.lower() == ".hjson"
    )
    if mode == "format":
        if is_hjson:
            raise ValueError(
                "--mode format requires strict JSON; use a config cleanup mode for HJSON"
            )
        return (
            reformat_json_text(
                text,
                indent=indent,
                max_inline=max_inline,
                sort_keys=sort_keys,
            )
            + "\n"
        )
    if is_hjson:
        import hjson

        payload = hjson.loads(text, object_pairs_hook=_unique_object)
    else:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    cleaned = cleanup_config(payload, mode=mode, base_config_path=str(source))
    return (
        json_dumps_streamlined(
            cleaned,
            indent=indent,
            max_inline=max_inline,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def discover_files(
    source: Path, *, max_depth: int = 1, include_hjson: bool = False
) -> list[Path]:
    if max_depth < 1:
        raise ValueError("--max-depth must be at least 1")
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(
            f"source does not exist or is not a file/directory: {source}"
        )
    suffixes = {".json", ".hjson"} if include_hjson else {".json"}
    found = []

    def visit(directory, depth):
        for path in sorted(directory.iterdir()):
            # A bulk selection must not escape its root through directory or
            # file symlinks. An explicitly selected single-file link is supported.
            if path.is_symlink():
                continue
            if path.is_file() and path.suffix.lower() in suffixes:
                found.append(path)
            elif path.is_dir() and depth < max_depth:
                visit(path, depth + 1)

    visit(source, 1)
    return sorted(found)


@dataclass
class PlannedFile:
    source: Path
    destination: Path
    original: bytes
    output: bytes


def _file_identity(path: Path):
    try:
        info = path.stat()
    except FileNotFoundError:
        inode = None
    else:
        inode = (info.st_dev, info.st_ino)
    return path.resolve(), inode


def plan_files(args) -> list[PlannedFile]:
    source = args.src.expanduser().absolute()
    destination = args.dst.expanduser().absolute() if args.dst else None
    files = discover_files(
        source, max_depth=args.max_depth, include_hjson=args.include_hjson
    )
    if not files:
        raise ValueError("no matching config files found")
    if source.is_dir() and destination is not None:
        src_root, dst_root = source.resolve(), destination.resolve()
        if src_root.is_relative_to(dst_root) or dst_root.is_relative_to(src_root):
            raise ValueError(
                "source and destination directories must not overlap; use --in-place explicitly"
            )
        if destination.exists() and not destination.is_dir():
            raise ValueError("a directory source requires a destination directory")

    # Index aliases once so large config collections do not require quadratic
    # scans and filesystem probes for every destination.
    source_ids = [_file_identity(path) for path in files]
    source_paths = {resolved for resolved, _ in source_ids}
    source_inodes = {inode for _, inode in source_ids if inode is not None}
    output_paths, output_inodes = set(), set()
    mappings = []
    for path in files:
        if destination is None:
            target = path.resolve()
        elif source.is_dir():
            target = destination / path.relative_to(source)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(
                    f"destination escapes output directory through a symlink: {target}"
                )
        else:
            target = destination
        resolved, inode = _file_identity(target)
        if destination is not None and (
            resolved in source_paths or inode is not None and inode in source_inodes
        ):
            raise ValueError(
                "destination must not overwrite a source; use --in-place explicitly"
            )
        if resolved in output_paths or inode is not None and inode in output_inodes:
            raise ValueError(f"multiple inputs map to the same destination: {target}")
        if target.is_symlink():
            raise ValueError(
                f"destination is a symlink: {target}; use --in-place on the source instead"
            )
        if target.exists():
            if not target.is_file():
                raise ValueError(f"destination is not a regular file: {target}")
            if destination is not None and not args.overwrite:
                raise FileExistsError(
                    f"destination already exists: {target}; use --overwrite"
                )
        mappings.append((path, target))
        output_paths.add(resolved)
        if inode is not None:
            output_inodes.add(inode)

    plans = []
    for path, target in mappings:
        try:
            original = path.read_bytes()
            output = render_source(
                original.decode("utf-8"),
                source=path,
                mode=args.mode,
                input_format=args.input_format,
                indent=args.indent,
                max_inline=args.max_inline,
                sort_keys=args.sort_keys,
            ).encode("utf-8")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        plans.append(PlannedFile(path, target, original, output))
    return plans


def atomic_write(path: Path, data: bytes, *, overwrite: bool) -> None:
    """Publish complete files, preserving existing ownership and permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.stat() if overwrite and path.exists() else None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if previous is not None:
            current = temporary.stat()
            if (current.st_uid, current.st_gid) != (previous.st_uid, previous.st_gid):
                if not hasattr(os, "chown"):
                    raise OSError("cannot preserve existing config ownership")
                os.chown(temporary, previous.st_uid, previous.st_gid)
            temporary.chmod(stat.S_IMODE(previous.st_mode))
        if overwrite:
            os.replace(temporary, path)
        else:
            # Linking a fully written sibling closes the no-clobber race without
            # exposing a partially written destination.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool clean-config",
        description="Clean/export configs offline, or format strict JSON without changing its values.",
    )
    parser.add_argument("src", type=Path, help="Source config file or directory")
    parser.add_argument(
        "dst",
        type=Path,
        nargs="?",
        help="Destination file or directory (required by default)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="full",
        help="full (default), live, backtest, optimize, or format only",
    )
    parser.add_argument(
        "--in-place", action="store_true", help="Explicitly replace source files"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing destination files",
    )
    parser.add_argument(
        "--max-depth",
        "--max_depth",
        type=int,
        default=1,
        help="Directory levels to scan: 1 = direct files, 2 = one subdirectory level (default: 1)",
    )
    parser.add_argument(
        "--include-hjson",
        action="store_true",
        help="Also select .hjson files in directory mode; output names/extensions are retained",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "json", "hjson"),
        default="auto",
        help="auto uses .hjson suffix, otherwise strict JSON",
    )
    parser.add_argument(
        "--indent", type=int, default=4, help="Indentation spaces (default: 4)"
    )
    parser.add_argument(
        "--max-inline",
        type=int,
        default=72,
        help="Maximum inline container length (default: 72)",
    )
    parser.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort object members in format mode; config cleanup modes always sort",
    )
    readonly = parser.add_mutually_exclusive_group()
    readonly.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list planned changes without writing; dst is optional",
    )
    readonly.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if any source would change (omit dst)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.in_place and args.dst is not None:
        parser.error("--in-place cannot be combined with dst")
    if args.check and args.dst is not None:
        parser.error("--check compares sources; omit dst")
    if args.dst is None and not (args.in_place or args.check or args.dry_run):
        parser.error("supply src and dst, or explicitly use --in-place")
    if args.overwrite and args.dst is None:
        parser.error("--overwrite requires dst; use --in-place to replace sources")
    if args.max_depth < 1 or args.indent < 0 or args.max_inline < 0:
        parser.error("--max-depth must be >= 1; --indent and --max-inline must be >= 0")
    try:
        # Preflight the complete batch before creating any output. Writes are
        # atomic per file, not a filesystem transaction across the directory.
        plans = plan_files(args)
        changed = sum(plan.original != plan.output for plan in plans)
        for plan in plans:
            different = plan.original != plan.output
            if args.check or args.dry_run:
                status = "Would change" if different else "Unchanged"
                print(f"{status}: {plan.source} -> {plan.destination}")
                continue
            if args.in_place:
                if not different:
                    print(f"Unchanged: {plan.source}")
                    continue
                if plan.destination.read_bytes() != plan.original:
                    raise ValueError(f"source changed after validation: {plan.source}")
            atomic_write(
                plan.destination, plan.output, overwrite=args.in_place or args.overwrite
            )
            print(f"Wrote: {plan.source} -> {plan.destination}")
        print(
            f"{len(plans)} file(s) checked; {changed} would change."
            if args.check or args.dry_run
            else f"{len(plans)} file(s) processed."
        )
        return 1 if args.check and changed else 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
