from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def main() -> int:
    output_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/run/passivbot/api-keys.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    user = require_env("PB_USER")
    payload = {
        user: {
            "exchange": require_env("PB_EXCHANGE"),
            "key": require_env("PB_API_KEY"),
            "secret": require_env("PB_API_SECRET"),
        }
    }

    passphrase = os.environ.get("PB_API_PASSPHRASE", "").strip()
    if passphrase:
        payload[user]["passphrase"] = passphrase

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
