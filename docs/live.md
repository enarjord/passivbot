# Running the bot live

## Fake Live Replay

Passivbot also includes a deterministic local `fake` exchange for replaying live-bot behavior without touching a real exchange.

Use it when you want to:

- reproduce a live issue with fixed candles and fills
- inspect HSL and mode transitions step by step
- regression-test the live loop locally after changing orchestration or exchange-facing code

Quick example:

```sh
PYTHONPATH=src python3 src/tools/run_fake_live.py \
  configs/fake_live_hsl_btc.hjson \
  scenarios/fake_live/hsl_long_red_restart.hjson \
  --user fake_hsl_restart_test \
  --snapshot-each-step
```

See [docs/fake_live.md](docs/fake_live.md) for scenario structure, included examples, outputs, and troubleshooting.

## Monitor Output

Passivbot can publish a disk-backed monitor root for external readers such as a future TUI/dashboard.

Current scope:

1. atomic current snapshot publication
2. structured NDJSON event stream
3. periodic compressed checkpoints

See [docs/monitor.md](docs/monitor.md) for config, output layout, and the currently emitted event kinds.


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
