## Quick Guide for Passivbot on Lighter

Lighter is a zero-fee, order-book-based decentralized perpetuals DEX (zkSync-based,
`mainnet.zklighter.elliot.ai`). Passivbot's `LighterBot` adapter uses USDC as the
quote currency and runs in one-way (non-hedge) mode only.

Unlike every other exchange Passivbot supports, Lighter requires a **platform-specific
native signing binary** in addition to the usual API credentials. This guide walks
through the extra step.

### Obtain Lighter API credentials

Follow the CCXT FAQ for creating an API key on Lighter and locating your account
index:

https://github.com/ccxt/ccxt/wiki/FAQ#how-to-use-the-lighter-exchange-in-ccxt

From that procedure you should end up with three values:

- `private_key` — the API key private key (hex, `0x`-prefixed). **Not** your
  wallet private key.
- `api_key_index` — the integer index (0–254) Lighter assigns to the API key.
- `account_index` — your Lighter account/sub-account index.

### Download the Lighter signer binary

Passivbot signs Lighter transactions through a compiled Go library that is loaded
at runtime via `ctypes`. CCXT ships the Python ctypes wrapper
(`ccxt.static_dependencies.lighter_client.signer`) but **not** the binary itself,
and the wrapper is pinned to a specific ABI. You must download a matching binary
and point Passivbot at it via `library_path`.

> **Use exactly the v1.0.4 / commit `b7fc10b2` binaries.**
> Newer releases of `elliottech/lighter-python` change the ctypes field layouts
> (for example `CreateOrderTxReq`) and CCXT's bundled Python wrapper has not been
> updated to match. Grabbing the latest release will break transaction signing.

1. Open the pinned directory in the upstream repo:
   https://github.com/elliottech/lighter-python/tree/b7fc10b2/lighter/signers
2. Download the file that matches your host platform (Linux/macOS/Windows,
   x86_64 or arm64). The repo ships one binary per platform in that directory.
3. Save the file somewhere persistent (for example
   `~/passivbot/lib/lighter-signer.so`). No `chmod +x` is needed — the file is
   loaded as a shared library, not executed.
4. If you run on Linux inside a minimal container and see
   `OSError: ... cannot open shared object file`, install `libc6` / `glibc` —
   the Go binary depends on it.

Only one binary needs to be loaded per Python process. If you run multiple
Lighter accounts in the same `passivbot` process, they will share the binary
that the first one loads.

### Passivbot setup

1. If not already installed, install Passivbot. Otherwise, pull the latest master
   branch and refresh dependencies:
   ```bash
   git pull
   python3 -m pip install -e .
   ```
2. Add a Lighter entry to `api-keys.json`:
   ```json
   "lighter_01": {
       "exchange": "lighter",
       "private_key": "0x_api_key_private_key_hex",
       "api_key_index": 0,
       "account_index": 0,
       "library_path": "/absolute/path/to/lighter-signer.so"
   }
   ```
   `library_path` should be an absolute path to the downloaded signer binary.
   A relative path works but is resolved against the process cwd at startup,
   so an absolute path is less error-prone.
3. Start the bot as usual:
   ```bash
   passivbot live -u lighter_01
   ```

### Operational notes

- **Quote currency is USDC.** Passivbot hardcodes this for Lighter; it will not
  trade USDT pairs there.
- **One-way mode only.** Lighter does not support hedge mode, so Passivbot runs
  with `hedge_mode = False`. Long and short positions on the same symbol cannot
  coexist — opening the opposite side will net against the existing position.
- **Native WebSocket.** `LighterBot` opens its own WebSocket to
  `wss://mainnet.zklighter.elliot.ai/stream` and subscribes to
  `account_all/{account_index}`, so order updates arrive via push rather than
  REST polling.
- **Per-symbol `fetch_open_orders`.** Lighter's REST API does not return
  all-symbols open orders in one call, so Passivbot fans out across tracked
  symbols and merges. This is handled automatically but means open-order
  refreshes scale with the number of active symbols.
- **Leverage/margin updates bypass CCXT.** `update_exchange_config_by_symbols`
  calls the signer's `SignUpdateLeverage` directly and submits via
  `publicPostSendTx`, because CCXT's own Lighter `load_account` path is broken
  against the current endpoint. If the exchange reports "no change" / "already",
  Passivbot treats it as a no-op and continues. Any other error is propagated
  so the orchestrator's per-symbol retry/backoff loop can handle it.
- **Startup failure modes.** If `library_path` is missing, unreadable, or wrong
  architecture, the bot fails loudly in `create_ccxt_sessions()` with
  `OSError` from `ctypes.CDLL`. If the signer loads but rejects your
  credentials, `CreateClient` returns an error string and Passivbot raises
  `lighter: CreateClient failed: ...` before any orders are placed.

### Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `OSError: ... cannot open shared object file` at startup | Wrong path, wrong architecture, or missing glibc in the runtime image. |
| `lighter: CreateClient failed: ...` | Wrong `private_key`, `api_key_index`, or `account_index`. Re-check them against the CCXT FAQ procedure linked above. |
| Transactions fail with signature/ABI errors | Binary is not the pinned v1.0.4 / `b7fc10b2` version. Re-download from the URL above. |
| Leverage/margin updates log "unchanged" forever | Expected when the symbol is already at the target leverage and margin mode; not an error. |
