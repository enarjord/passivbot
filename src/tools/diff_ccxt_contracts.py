import argparse
import json

from ccxt_contracts import DEFAULT_DIFF_IGNORE_PATHS, diff_snapshots, load_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diff two CCXT contract snapshots captured for Passivbot."
    )
    parser.add_argument("old_snapshot", help="Baseline snapshot JSON")
    parser.add_argument("new_snapshot", help="Candidate snapshot JSON")
    parser.add_argument(
        "--ignore",
        default=",".join(sorted(DEFAULT_DIFF_IGNORE_PATHS)),
        help="Comma-separated flattened paths to ignore",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of per-path changes to print for each bucket",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON diff instead of a readable summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ignore_paths = [x.strip() for x in args.ignore.split(",") if x.strip()]
    diff = diff_snapshots(
        load_snapshot(args.old_snapshot),
        load_snapshot(args.new_snapshot),
        ignore_paths=ignore_paths,
    )
    if args.json:
        print(json.dumps(diff, indent=2, sort_keys=True))
        return

    print(
        "summary:",
        f"added={diff['summary']['added']}",
        f"removed={diff['summary']['removed']}",
        f"changed={diff['summary']['changed']}",
    )
    for bucket in ("added", "removed", "changed"):
        items = diff[bucket]
        if not items:
            continue
        print(f"\n{bucket}:")
        for item in items[: args.limit]:
            if bucket == "added":
                print(f"  + {item['path']} = {item['new']!r}")
            elif bucket == "removed":
                print(f"  - {item['path']} = {item['old']!r}")
            else:
                print(f"  ~ {item['path']}: {item['old']!r} -> {item['new']!r}")
        remaining = len(items) - args.limit
        if remaining > 0:
            print(f"  ... {remaining} more")


if __name__ == "__main__":
    main()
