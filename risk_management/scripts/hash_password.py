#!/usr/bin/env python3
"""Utility script to create bcrypt password hashes for the web dashboard."""

from __future__ import annotations

import argparse
import getpass

from passlib.context import CryptContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a bcrypt hash for dashboard users")
    parser.add_argument("password", nargs="?", help="Optional password to hash. If omitted a prompt is used.")
    args = parser.parse_args(argv)

    password = args.password or getpass.getpass("Password: ")
    if not password:
        parser.error("Password cannot be empty")
    context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    print(context.hash(password))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
