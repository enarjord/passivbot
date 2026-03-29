# Running the bot live

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
python src/passivbot.py --custom-endpoints configs/custom_endpoints.json configs/your_config.hjson

# Disable overrides even if the file exists
python src/passivbot.py --custom-endpoints none configs/your_config.hjson
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
