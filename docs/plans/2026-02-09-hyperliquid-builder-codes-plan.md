# Hyperliquid Builder Codes: Research Summary + Implementation Plan

Date: 2026-02-09

## Scope and constraints

Requested constraints:

- Keep implementation simple and aligned with existing Passivbot patterns (`broker_codes.hjson`, CCXT options).
- Enable by default.
- Do not block trading when builder fee approval is missing.
- If builder code cannot be used (common with API/agent wallets), print a large recurring banner encouraging approval (similar spirit to Binance `print_new_user_suggestion`).
- Verify CCXT built-in support and whether current Passivbot version supports it.
- Recommend a practical default fee level.

---

## Verified findings (official docs + code)

### Hyperliquid protocol behavior

- Builder fees are attached per order with `builder: {"b": <address>, "f": <int>}`.
- Fee unit is tenths of a basis point (`f=10` means 1 bps = 0.01%).
- Maximum fee caps:
  - Perps: 0.1% (10 bps)
  - Spot: 1% (100 bps)
- Builder account requirement: at least 100 USDC account value.
- Approval is required via `approveBuilderFee` action.
- Approval status is queryable via info endpoint `{"type":"maxBuilderFee", "user":..., "builder":...}`.

### Agent/main wallet constraint

- Official Hyperliquid Python SDK example explicitly raises if wallet is not the main wallet before approval call.
- This confirms approval signing is main-wallet scoped, while many Passivbot users run trading from API/agent wallets.

### CCXT built-in behavior

- CCXT Hyperliquid has built-in builder handling:
  - `handle_builder_fee_approval()`
  - `approve_builder_fee()`
  - `orderAction['builder'] = {'b': ..., 'f': feeInt}` when `approvedBuilderFee` is true
- Defaults include:
  - default builder address (CCXT’s own)
  - default `feeRate = "0.01%"`
  - default `feeInt = 10` (1 bps)
- In current installed CCXT (`4.5.22`), client initialization also runs referral setup via `set_ref()`.

---

## CCXT version support check

I verified version behavior by inspecting `ccxt/async_support/hyperliquid.py` in released versions.

| CCXT version | Built-in builder handling present? | Evidence |
|---|---:|---|
| 4.4.93 | No | no `handle_builder_fee_approval` / `approve_builder_fee` |
| 4.4.94 | Yes | methods + `orderAction['builder']` path appear |
| 4.4.100 | Yes | same builder handling present |
| 4.5.22 | Yes | present in installed Passivbot environment |

Conclusion: builder-code automation appears from **CCXT 4.4.94+**.  
Passivbot currently pins **`ccxt==4.5.22`**, so version support is already sufficient.

---

## Current Passivbot state (relevant to implementation)

- `broker_codes.hjson` currently has no `hyperliquid` entry.
- `self.broker_code` is loaded for all exchanges, including Hyperliquid, but Hyperliquid does not currently apply builder/ref options.
- `src/exchanges/hyperliquid.py:create_ccxt_sessions()` currently does not set Hyperliquid builder/ref options.
- Therefore CCXT defaults are in effect today (CCXT referral/builder, not Passivbot’s own).
- No Hyperliquid-specific recurring suggestion banner currently exists.

---

## Differences vs Claude summary

Mostly aligned, with these verified clarifications:

1. **Exact CCXT cutoff**: built-in support starts at **4.4.94**, not just “some 4.x”.
2. **Fee integer consistency**: for a 1 bps default, use **`feeInt: 10`** (not `1`).
3. **Agent-wallet operational nuance (inference from docs + CCXT flow)**:
   - CCXT attempts approval signing from the active wallet.
   - If that wallet lacks approval permission, CCXT disables builder fee silently.
   - To keep behavior robust/non-blocking and user-visible, Passivbot should add explicit status checks + recurring banner.

---

## Recommended default fee level

Recommended initial default for Passivbot:

- `feeRate: "0.01%"`
- `feeInt: 10` (1 bps)

Rationale:

- Far below perps max cap (10 bps), so not greedy.
- Meaningful enough to avoid “too low” dust economics.
- Matches current CCXT default semantics, reducing surprise.

---

## Implementation plan (simple, non-blocking, default-on)

### 1) Add Hyperliquid builder config to broker codes

File: `broker_codes.hjson`

Add:

```hjson
hyperliquid: {
    ref: "PASSIVBOT"
    builder: "0x<passivbot_builder_wallet>"
    feeRate: "0.01%"
    feeInt: 10
    builderFee: true
}
```

### 2) Apply builder/ref options in Hyperliquid CCXT sessions

File: `src/exchanges/hyperliquid.py`

In `create_ccxt_sessions()`:

- Read `self.broker_code`.
- If it is a dict, map keys directly into `client.options` for `cca` and `ccp`.
- Keep this override minimal and explicit to match existing broker-code pattern.
- Continue non-blocking behavior if keys are missing/malformed (log warning, do not raise).

### 3) Add recurring “approve builder code” banner (non-blocking)

File: `src/exchanges/hyperliquid.py`

Add a method similar to Binance’s suggestion printer:

- Throttle with interval (`x` minutes; recommended 60 minutes).
- Use a large boxed banner.
- Message should clearly say:
  - trading continues normally,
  - builder attribution is currently inactive,
  - one-time approval must be done from main wallet,
  - then bot can continue using API wallet.

### 4) Detect whether builder attribution is active

Use lightweight checks to decide when to print banner:

- Primary signal: CCXT options state (`approvedBuilderFee` true => active).
- If inactive, query `maxBuilderFee` via Hyperliquid info endpoint for configured `user` + `builder`.
- If approval is absent/insufficient, print recurring banner.
- Do not block order creation/cancellation in any branch.

Implementation note (inference): this avoids silent failure UX when CCXT approval signing fails on agent wallets.

### 5) Hook banner into runtime loop

File: `src/exchanges/hyperliquid.py`

- Override `execute_to_exchange()` in Hyperliquid bot (same pattern as Binance).
- Call `await super().execute_to_exchange()` then banner check/print.

### 6) Documentation updates

- Add short user-facing setup section to `docs/hyperliquid_guide.md`:
  - one-time main wallet approval
  - API/agent wallet can still trade if not approved (with periodic reminder)
- Update/replace `docs/hyperliquid_builder_codes.md` sections that currently suggest per-order param injection in `_build_order_params` (prefer CCXT option-based flow).

### 7) Validation checklist

- Unit test parsing of approval response (`"0"`, `"0.01%"`, etc.).
- Mocked Hyperliquid exchange test:
  - approval missing -> orders still execute + banner throttles.
  - approval present -> no banner after activation.
- Regression check: no behavior changes for non-Hyperliquid exchanges.

---

## Open inputs needed before implementation

1. Final Passivbot builder wallet address (`builder`).
2. Confirm banner interval `x` minutes (recommended: 60).
3. Confirm default fee stays at 1 bps (`feeInt=10`).

---

## Sources

- Hyperliquid Builder Codes: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/builder-codes
- Hyperliquid Exchange Endpoint (`approveBuilderFee` action): https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint
- Hyperliquid Info Endpoint (`maxBuilderFee` query): https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint
- Hyperliquid Python SDK builder example: https://raw.githubusercontent.com/hyperliquid-dex/hyperliquid-python-sdk/master/examples/basic_builder_fee.py
- CCXT Hyperliquid (v4.4.93): https://raw.githubusercontent.com/ccxt/ccxt/4.4.93/python/ccxt/async_support/hyperliquid.py
- CCXT Hyperliquid (v4.4.94): https://raw.githubusercontent.com/ccxt/ccxt/4.4.94/python/ccxt/async_support/hyperliquid.py
- CCXT Hyperliquid (v4.4.100): https://raw.githubusercontent.com/ccxt/ccxt/4.4.100/python/ccxt/async_support/hyperliquid.py
