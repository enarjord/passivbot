# Running the bot live

## Logging

`passivbot live` now writes its console log stream to a timestamped file under `logs/` by default.
It also refreshes `logs/{user}.log` as a stable alias to the current run's logfile, which is useful
for monitor tooling that wants a fixed path per bot. This behavior is controlled from
`config.logging`:

- `logging.persist_to_file = true` keeps on-disk live logs enabled
- `logging.dir` chooses the target directory for both the timestamped archive and the stable alias
- `logging.rotation`, `logging.max_bytes_mb`, and `logging.backup_count`
  control optional file rotation

Disable file persistence only if you explicitly want console-only live logging:

```json
"logging": {
  "persist_to_file": false
}
```

## Concurrent Passivbot Protection

The live bot now watches for newer open orders that look like Passivbot-managed orders but were not
emitted by the current runtime. Manual/non-Passivbot orders are ignored, and older inherited orders
are ignored as well.

If this detection fires repeatedly within its rolling window, the bot stops itself to avoid two
Passivbot instances silently competing on the same account.

If you see this stop condition, first check for:

- another Passivbot process on the same machine (`tmux`, `screen`, background shell, `systemd`, etc.)
- another host or VPS using the same exchange user / API key
- an old bot instance that was restarted elsewhere and is still running

## Custom Exchange Endpoints

Some integrations require routing REST traffic through an intermediate service
before it reaches the exchange. Passivbot supports this via a JSON configuration
file located at `configs/custom_endpoints.json`.

### 1. Create the config file

Copy the example file and edit it:

```sh
cp custom_endpoints.json.example configs/custom_endpoints.json
```

Populate the file with the proxy endpoints. A minimal Binance USDT‑margined
futures override might look like:

```json
{
  "defaults": {
    "disable_ws": false,
    "rest": {
      "rewrite_domains": {},
      "url_overrides": {},
      "extra_headers": {}
    }
  },
  "exchanges": {
    "binanceusdm": {
      "disable_ws": true,
      "rest": {
        "rewrite_domains": {
          "https://fapi.binance.com": "https://proxy.example.exchange"
        }
      }
    }
  }
}
```

- `rewrite_domains` swaps the original host while preserving the API path. Any
  ccxt URL starting with `https://fapi.binance.com` will be routed through the proxy.
- `disable_ws` prevents websocket initialisation if the proxy does not provide
  matching websocket endpoints.

### 2. Start the bot

When `configs/custom_endpoints.json` exists, the overrides are loaded automatically.
The CLI also accepts an explicit path or can disable overrides entirely:

```sh
# Use custom endpoints from a specific file
passivbot live --custom-endpoints configs/custom_endpoints.json configs/your_config.hjson

# Disable overrides even if the file exists
passivbot live --custom-endpoints none configs/your_config.hjson
```

You can embed the path in your live config so it is always applied:

```json
"live": {
  "custom_endpoints_path": "configs/custom_endpoints.json",
  ...
}
```

### 3. Confirm during startup

Passivbot logs every override at launch. Look for entries such as:

```
INFO Custom endpoint active for binanceusdm.fapiPrivate: https://fapi.binance.com/fapi/v1 -> https://proxy.example.exchange/fapi/v1
INFO Custom endpoint override active for binanceusdm (disable_ws=True, source=...)
```

If the proxy rejects the request (e.g., missing credentials) ccxt will report a
403/401 with the proxied URL, confirming the routing is in effect.

> **Note:** The override file only affects REST endpoints. Websocket sessions are
> skipped when `disable_ws` is `true`; otherwise Passivbot continues to connect
> directly to the exchange websocket hosts. When websockets are disabled the bot
> still refreshes orders and positions on its normal schedule via REST polling,
> so trading continues—only the `watch_orders` push updates are suppressed.
