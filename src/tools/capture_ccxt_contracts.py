import argparse
import asyncio

from ccxt_contracts import (
    DEFAULT_CAPTURE_SECTIONS,
    capture_contract_snapshot,
    default_snapshot_path,
    dump_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture live CCXT contract snapshots for Passivbot upgrade checks."
    )
    parser.add_argument("--user", required=True, help="api-keys.json user to capture with")
    parser.add_argument(
        "--label",
        default=None,
        help="Friendly label used in metadata and default output filename (default: user)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to artifacts/ccxt_contracts/{exchange}/{label}.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ccxt_contracts",
        help="Base directory for default output paths",
    )
    parser.add_argument(
        "--sections",
        default=",".join(DEFAULT_CAPTURE_SECTIONS),
        help=f"Comma-separated sections to capture (default: {','.join(DEFAULT_CAPTURE_SECTIONS)})",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbols for symbol-scoped order/trade capture",
    )
    parser.add_argument(
        "--trades-limit",
        type=int,
        default=25,
        help="Limit for raw trade capture when trades section is enabled",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    sections = [x.strip() for x in args.sections.split(",") if x.strip()]
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    snapshot = await capture_contract_snapshot(
        user=args.user,
        label=args.label,
        sections=sections,
        symbols=symbols,
        trades_limit=args.trades_limit,
    )
    output_path = args.output
    if output_path is None:
        output_path = default_snapshot_path(
            args.output_dir,
            snapshot["meta"]["exchange"],
            args.label or args.user,
        )
    dumped = dump_snapshot(snapshot, output_path)
    print(dumped)


if __name__ == "__main__":
    asyncio.run(main())
