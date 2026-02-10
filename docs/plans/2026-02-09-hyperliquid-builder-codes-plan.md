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

Additional required behavior updates (2026-02-10):

- Builder codes must be **on by default**.
- If approval already exists: print a **small thank-you banner**.
- If approval is missing and user runs with **main wallet key**: auto-approve builder fee at startup, print notice banner, then thank-you banner.
- If approval is missing and user runs with **agent/API wallet**:
  - allow first order attempt with builder enabled (may fail with missing approval error),
  - then temporarily disable builder usage for that account,
  - periodically (about every 30 minutes) re-enable and retry,
  - on failure, print a large instruction banner with clear approval paths.
- User should get clear approval guidance **well before first live run** (docs + config examples and optionally startup/install surfaces).

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
- Practical implication: without explicit override, CCXT may attempt builder approval for its own default builder address.

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

### Why 1 bps over 2 bps (for default)

Both are technically viable. This plan recommends **1 bps default** for rollout safety:

- **Lower friction at first-run**: builder codes are default-on and include nag/error UX for unapproved users; 1 bps reduces user resistance during adoption.
- **Closer to existing CCXT expectations**: CCXT’s built-in defaults are already aligned with 1 bps, which minimizes surprise and “hidden fee jump” perception.
- **Easier trust ramp for open-source users**: start conservative, then revisit after adoption/retention data.
- **Simple upgrade path**: if community response is positive, moving default to 2 bps later is a one-line `broker_codes.hjson` change plus changelog note.

Practical policy suggestion:

- Start with `feeInt=10` (1 bps) for one release cycle.
- Re-evaluate with observed approval rates / user feedback.
- Consider `feeInt=20` (2 bps) only after communication + release-note notice.

---

## Implementation plan (simple, non-blocking, default-on)

### 1) Add Hyperliquid builder config to broker codes

File: `broker_codes.hjson`

Add:

```hjson
hyperliquid: {
    ref: "PASSIVBOT"
    builder: "0x5e20A6D7e11366390Fde63EA3d0A026903359a74"
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
- Set `builderFee=False` and `approvedBuilderFee=False` initially so passivbot controls approval flow explicitly (instead of relying on CCXT auto-approval timing/behavior).

### 3) Add recurring “approve builder code” banner (non-blocking)

File: `src/exchanges/hyperliquid.py`

Add methods similar to Binance’s suggestion printer with two tiers:

- **Small thank-you banner** (only when active/approved): concise support acknowledgment.
- **Large warning banner** (when approval missing and builder cannot currently be used):
  - explain why it failed,
  - keep bot running,
  - list clear approval options:
    1) approve in Hyperliquid web UI,
    2) temporarily use main wallet key in `api-keys.json` so bot can auto-approve,
    3) use planned CLI helper in `src/tools/`.

### 4) Detect whether builder attribution is active

Use lightweight checks to decide when to print banner:

- Primary signal: CCXT options state (`approvedBuilderFee` true => active).
- If inactive, query `maxBuilderFee` via Hyperliquid info endpoint for configured `user` + `builder`.
- If approval is absent/insufficient, print recurring banner.
- Do not block order creation/cancellation in any branch.

Implementation note (inference): this avoids silent failure UX when CCXT approval signing fails on agent wallets.

### 5) Startup behavior by wallet mode (required)

At startup, when builder is enabled and not yet approved:

- Run one explicit initializer (e.g. `_init_builder_codes()`) once from the first `execute_to_exchange()` cycle.

- **Main wallet mode** (`exchange.account_address == exchange.wallet.address` equivalent):
  - call approval automatically via CCXT/Hyperliquid API,
  - print: “builder codes approved to <address>, fee <rate>”,
  - print small thank-you banner.
- **Agent/API wallet mode**:
  - do not hard-stop startup,
  - force one builder-attributed order attempt by temporarily setting `approvedBuilderFee=True`,
  - allow first builder-attributed order attempt to fail naturally with Hyperliquid error,
  - immediately switch account session to temporary builder-disabled mode.

### 6) Temporary disable + scheduled re-enable loop (required)

For agent/API wallet accounts with missing approval:

- After first approval-related failure, set a per-account cooldown state.
- During cooldown, place normal orders without builder params.
- Every ~30 minutes:
  - re-enable builder attempt,
  - if still unapproved, allow visible error and print large instruction banner again,
  - re-enter cooldown.
- If approval later succeeds, keep builder enabled and switch to thank-you banner flow.
- Suggested state variables:
  - `_builder_initialized`
  - `_builder_pending_approval`
  - `_builder_disabled_ts`
  - `_builder_next_retry_ts`

### 7) Hook banner and recheck flow into runtime loop

File: `src/exchanges/hyperliquid.py`

- Override `execute_to_exchange()` in Hyperliquid bot (same pattern as Binance).
- Call `await super().execute_to_exchange()` then:
  - evaluate builder state transitions,
  - print small/large banner as needed,
  - run cooldown re-enable checks on schedule.

### 8) Early user guidance in docs/setup surfaces (required)

- Add explicit pre-live warning to `docs/hyperliquid_guide.md` near setup steps:
  - approval must happen once,
  - main wallet can auto-approve,
  - agent wallet requires prior manual approval.
- Add clear builder-approval note to `api-keys.json.example` around Hyperliquid credentials.
- Add concise “before first live start” note in installation/onboarding docs where practical.
- Update/replace `docs/hyperliquid_builder_codes.md` sections that currently suggest per-order param injection in `_build_order_params` (prefer CCXT option-based flow).
- Add planned CLI approval helper docs for `src/tools/` once tool exists.
- Add `src/tools/approve_builder_fee.py` plan details:
  - load builder/fee from `broker_codes.hjson`,
  - accept main wallet creds once,
  - call `approve_builder_fee`,
  - print success/failure guidance.

### 9) Validation checklist

- Unit test parsing of approval response (`"0"`, `"0.01%"`, etc.).
- Mocked Hyperliquid exchange test:
  - approval missing -> orders still execute + banner throttles.
  - approval present -> no banner after activation.
- Mocked main-wallet startup test:
  - unapproved -> auto-approve called -> notice + thank-you.
- Mocked agent-wallet flow test:
  - first failure visible,
  - builder temporarily disabled,
  - 30-minute re-enable retry logic works.
- Regression check: no behavior changes for non-Hyperliquid exchanges.

---

## Risk notes

- Expected tradeoff for agent-wallet nag path: one order attempt may fail each retry cycle before fallback disables builder again.
- This is intentional for visibility and should be documented as part of the “free-with-ads” UX.

---

## Fixed implementation inputs

1. Builder wallet address: `0x5e20A6D7e11366390Fde63EA3d0A026903359a74`
2. Default fee: `feeRate="0.01%"`, `feeInt=10` (1 bps)
3. Agent-wallet retry cadence target: ~30 minutes

---

## Sources

- Hyperliquid Builder Codes: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/builder-codes
- Hyperliquid Exchange Endpoint (`approveBuilderFee` action): https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint
- Hyperliquid Info Endpoint (`maxBuilderFee` query): https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint
- Hyperliquid Python SDK builder example: https://raw.githubusercontent.com/hyperliquid-dex/hyperliquid-python-sdk/master/examples/basic_builder_fee.py
- CCXT Hyperliquid (v4.4.93): https://raw.githubusercontent.com/ccxt/ccxt/4.4.93/python/ccxt/async_support/hyperliquid.py
- CCXT Hyperliquid (v4.4.94): https://raw.githubusercontent.com/ccxt/ccxt/4.4.94/python/ccxt/async_support/hyperliquid.py
- CCXT Hyperliquid (v4.4.100): https://raw.githubusercontent.com/ccxt/ccxt/4.4.100/python/ccxt/async_support/hyperliquid.py
- Supplemental implementation context: https://github.com/freqtrade/freqtrade/issues/11986
