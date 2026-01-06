import argparse
import asyncio
import os
import platform
import sys


if __name__ == "__main__":
    min_version = (3, 12)
    if sys.version_info < min_version:
        current = ".".join(map(str, sys.version_info[:3]))
        required = ".".join(map(str, min_version))
        print(
            "\n".join(
                [
                    f"Passivbot requires Python {required} (earlier versions are not supported).",
                    f"You are running Python {current} from: {sys.executable}",
                    "",
                    "Fix:",
                    f"  - Install Python {required}",
                    f"  - Recreate your venv with: python{required} -m venv venv",
                    "  - Reinstall deps and try again",
                    "",
                    "See docs/installation.md for full instructions.",
                ]
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    from rust_utils import check_and_maybe_compile

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
