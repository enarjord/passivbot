import argparse
import asyncio
import os
import platform
import sys

from rust_utils import check_and_maybe_compile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip-rust-compile", action="store_true", help="Skip Rust build check.")
    parser.add_argument(
        "--force-rust-compile", action="store_true", help="Force rebuild of Rust extension."
    )
    parser.add_argument(
        "--fail-on-stale-rust",
        action="store_true",
        help="Abort if Rust extension appears stale instead of attempting rebuild.",
    )
    known_args, remaining = parser.parse_known_args()

    try:
        check_and_maybe_compile(
            skip=known_args.skip_rust_compile
            or os.environ.get("SKIP_RUST_COMPILE", "").lower() in ("1", "true", "yes"),
            force=known_args.force_rust_compile,
            fail_on_stale=known_args.fail_on_stale_rust,
        )
    except Exception as exc:
        print(f"Rust extension check failed: {exc}")
        sys.exit(1)

    # Recreate argv for the real app without the rust flags
    sys.argv = [sys.argv[0]] + remaining
    from passivbot import main

    asyncio.run(main())
