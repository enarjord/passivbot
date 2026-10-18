"""Retired v8 HSL migration entry point; v9 requires explicit memory choices."""

from argparse import ArgumentParser


MESSAGE = (
    "migrate-hsl targets the retired v8 HSL policy. Use passivbot tool migrate-config-v9 "
    "with --lookback-ema-span-minutes, review --dry-run --diff, and acknowledge any "
    "removed Never policy explicitly."
)


def migrate(*args, **kwargs):
    raise ValueError(MESSAGE)


def main(argv=None):
    parser = ArgumentParser(prog="passivbot tool migrate-hsl", description=MESSAGE)
    parser.error(MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
