"""Lighter existing-key configuration shared by trading and balance tools."""

import hashlib
import re
from pathlib import Path

# CCXT 4.5.66's ctypes ABI matches this official lighter-python revision.
SIGNER_REVISION = "8bac9f56b9d0dd0eedaeb53a00ccb4fc9d77082e"
SIGNER_SHA256 = {
    "linux-amd64": "28d27620a648510e826250707246fca84d299df59080385cc6649c601e94944c",
    "linux-arm64": "a56b761531dc9242456993c45739115fa235dd1ad095cb3b27c30e4d074db3ad",
    "darwin-arm64": "581a8416fbfdd1196c21fb8e3a9af1dd39a8d5f2c65dce6e9d5abf2319fc8919",
    "windows-amd64": "b1ffc4bdaefa595112f3429cab9536fe92d6e6bdd52057b6faedeb78cb2975b0",
}


def client_config(user: dict) -> dict:
    """Do not pass an L2 key as CCXT's L1 privateKey (which rotates keys)."""
    account = user["account_index"]
    key_index = user["api_key_index"]
    if isinstance(account, bool) or not isinstance(account, int) or account < 0:
        raise ValueError("Lighter account_index must be a nonnegative integer")
    if (
        isinstance(key_index, bool)
        or not isinstance(key_index, int)
        or not 4 <= key_index <= 254
    ):
        raise ValueError("Lighter api_key_index must be an integer between 4 and 254")
    private_key = user["private_key"].removeprefix("0x")
    if not re.fullmatch(r"[0-9a-fA-F]{80}", private_key):
        raise ValueError("Lighter private_key must be an L2 API private key")
    options = dict(user.get("options", {}))
    path = str(Path(user["signer_path"]).expanduser().resolve())
    if not Path(path).is_file():
        raise ValueError(
            "Lighter signer_path must identify the compatible official signer"
        )
    if (
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        not in SIGNER_SHA256.values()
    ):
        raise ValueError(
            f"Lighter signer must match official revision {SIGNER_REVISION}"
        )
    options.update(
        {
            "accountIndex": account,
            "apiKeyIndex": key_index,
            "libraryPath": path,
            "builderFee": False,
            "defaultType": "swap",
            "auths": {
                str(account): {
                    str(key_index): {
                        "signer": None,
                        "lighterPrivateKey": private_key,
                        "deadline": None,
                        "token": None,
                    }
                }
            },
        }
    )
    return {"enableRateLimit": True, "timeout": 30000, "options": options}
