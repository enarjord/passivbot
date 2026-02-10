# Hyperliquid Builder Codes - Research & Implementation Plan

## Research Summary

### What Are Builder Codes?

Builder codes are an **onchain, per-order attribution mechanism** on Hyperliquid that lets
applications earn fees on orders they route for users. Key properties:

- **Per-order**: Builder info is attached to each individual order action as `{"b": "0xAddress", "f": feeInt}`
- **Fee unit**: `f` is in **tenths of a basis point** (value `10` = 1 bps = 0.01%)
- **Limits**: Max 10 bps (0.1%) on perps, max 100 bps (1%) on spot
- **Revenue**: 100% goes to the builder. Over $40M paid out ecosystem-wide to date
- **Builder requirement**: Builder wallet must hold >= 100 USDC in perps account

### Approval Flow

Users must **one-time approve** a builder's maximum fee rate via the `ApproveBuilderFee` action.

**Critical constraint**: This approval **must be signed with the main wallet**, not an agent/API wallet.
This is the single most important implementation detail - most passivbot users run with agent wallets,
so they cannot auto-approve from inside the bot. They must approve externally (e.g. via Hyperliquid
web UI, SDK script, or a separate passivbot utility).

Once approved, the user can revoke at any time. The bot (even with an agent wallet) can then
attach builder info to every order - only the initial approval requires the main wallet.

### Querying Approval Status

```
POST https://api.hyperliquid.xyz/info
{"type": "maxBuilderFee", "user": "0xUSER", "builder": "0xBUILDER"}
```

Returns the maximum fee the user has approved for the builder. If 0 or absent, not approved.

### CCXT Built-in Support

CCXT (>= ~4.4.80, exact version unclear) has **full built-in builder code handling** in its
Hyperliquid module. The installed passivbot version (CCXT 4.5.22) **fully supports this**.

#### How It Works in CCXT

On `initialize_client()` (called automatically on first `fetch_markets`, `create_order`, etc.):

1. **`set_ref()`** - Sets referral code via `setReferrer` action. Default: `'CCXT1'`
2. **`handle_builder_fee_approval()`** - Attempts to approve builder fee. Default builder: `0x6530512A6c89C7cfCEbC3BA7fcD9aDa5f30827a6` (CCXT's own address)

On every subsequent order, if `approvedBuilderFee` is true in options, CCXT automatically
attaches `builder: {"b": address, "f": feeInt}` to the order action.

#### CCXT Options

| Option | Default | Purpose |
|--------|---------|---------|
| `ref` | `'CCXT1'` | Referral code for `setReferrer` |
| `builder` | `'0x6530...'` (CCXT) | Builder wallet address |
| `feeRate` | `'0.01%'` | Max fee rate string for approval tx |
| `feeInt` | `10` (= 1 bps) | Fee in tenths of bps, attached to orders |
| `builderFee` | `True` | Master enable/disable toggle |
| `approvedBuilderFee` | `False` | Set to true after successful approval |
| `refSet` | `False` | Set to true after referral is set |

#### Current State: CCXT's Builder Code Is Already Active

**Important discovery**: With CCXT 4.5.22, CCXT is **already attempting** to approve its own
builder address on every passivbot Hyperliquid session. If the user's main wallet has
previously approved CCXT's builder (e.g. via another CCXT-based tool), CCXT charges 1 bps
on every order. If approval fails (agent wallet), it silently disables and no builder fee
is charged.

This was confirmed by the [freqtrade issue #11986](https://github.com/freqtrade/freqtrade/issues/11986)
where users on CCXT 4.4.94 hit "Builder fee has not been approved" errors, while CCXT 4.4.77
did not have this behavior.

### CCXT Version Compatibility

| CCXT Version | Builder Fee Support |
|---|---|
| < 4.4.77 | No builder fee support |
| 4.4.77-4.4.93 | Transition period (unclear) |
| >= 4.4.94 | Full builder fee with `initialize_client()` |
| **4.5.22** (passivbot current) | **Fully supported** |

### Hyperliquid Base Fee Context

To recommend a builder fee, context on Hyperliquid's base fees:

| Tier | 14d Volume | Perp Taker | Perp Maker |
|------|------------|------------|------------|
| Base | $0 | 0.045% (4.5 bps) | 0.010% (1 bps) |
| Tier 2 | $5M | 0.040% | 0.008% |
| Tier 3 | $25M | 0.035% | 0.006% |
| Tier 4 | $100M | 0.030% | 0.000% |

Passivbot primarily places **maker** (post-only) orders, so users pay 0.5-1 bps in fees.
A builder fee adds on top of these exchange fees.

### Fee Level Recommendation

**Recommended default: 2 bps (0.02%) = `feeInt: 20`**

Rationale:
- CCXT's own default is 1 bps (`feeInt: 10`). We should be at or above this since passivbot
  provides significantly more value than a library wrapper.
- Phantom wallet charges builder fees and earns ~$100k/day. PVP.trade has earned $7.2M lifetime.
  These are consumer-facing UIs.
- HeyAnon charges 10 bps (0.10%) on profitable position closes.
- Passivbot is an open-source project; 2 bps is modest and fair.
- For a typical passivbot user doing $1M monthly volume: 2 bps = ~$200/month to the project.
- At base tier with maker orders (1 bps exchange fee), a 2 bps builder fee means the user
  pays 3 bps total vs 1 bps without. This is still far below taker fees (4.5 bps).
- Users can easily edit `broker_codes.hjson` to adjust if desired.

Alternative considered: 1 bps (`feeInt: 10`) - matches CCXT default, but undersells passivbot's value.

### Differences from GPT's Research

1. **GPT suggested off by default** (`hyperliquid_builder_enabled = false`). User requirement is
   **on by default** with a nag banner for unapproved users.

2. **GPT suggested a new config key** (`live.hyperliquid_builder_enabled`). This is unnecessary -
   the existing `broker_codes.hjson` pattern handles this cleanly, and CCXT options control the behavior.

3. **GPT suggested explicitly disabling CCXT's default behavior**. This is correct in spirit -
   we should override CCXT's default builder address with passivbot's, not add on top.

4. **GPT suggested approval-status check via raw HTTP**. We can use CCXT's built-in
   `handle_builder_fee_approval()` which does this automatically, plus we can query
   `maxBuilderFee` for the banner logic.

5. **GPT's approval flow is mostly correct** but misses that CCXT handles the signing and
   submission automatically via `initialize_client()` -> `approve_builder_fee()`.

---

## Implementation Plan

### Overview

Use CCXT's built-in builder code infrastructure. Override CCXT's default builder address with
passivbot's in `create_ccxt_sessions()`. Add a periodic nag banner (like Binance's
`print_new_user_suggestion`) that checks approval status and prints a large message encouraging
users to approve the builder code.

### Prerequisites

- A Hyperliquid wallet address controlled by the passivbot project, funded with >= 100 USDC
  in the perps subaccount. This becomes the builder address in `broker_codes.hjson`.

### File Changes

#### 1. `broker_codes.hjson` - Add Hyperliquid builder config

```hjson
hyperliquid: {
    ref: "PASSIVBOT"
    builder: "0xYOUR_PASSIVBOT_BUILDER_ADDRESS"
    feeRate: "0.02%"
    feeInt: 20
}
```

- `ref`: Referral code (replaces CCXT's default `CCXT1`)
- `builder`: Passivbot's builder wallet address (replaces CCXT's default address)
- `feeRate`: Max fee rate string for the approval transaction
- `feeInt`: 20 = 2 bps, attached to every order

#### 2. `src/exchanges/hyperliquid.py` - Wire builder codes + nag banner

**In `create_ccxt_sessions()`**, after existing options setup:

```python
# Configure builder code and referral
if isinstance(self.broker_code, dict):
    hl_opts = {}
    if "ref" in self.broker_code:
        hl_opts["ref"] = self.broker_code["ref"]
    if "builder" in self.broker_code:
        hl_opts["builder"] = self.broker_code["builder"]
        hl_opts["feeRate"] = self.broker_code.get("feeRate", "0.02%")
        hl_opts["feeInt"] = self.broker_code.get("feeInt", 20)
    for client in [c for c in [self.cca, getattr(self, 'ccp', None)] if c]:
        client.options.update(hl_opts)
elif self.broker_code:
    # Simple string = referral code only
    for client in [c for c in [self.cca, getattr(self, 'ccp', None)] if c]:
        client.options["ref"] = self.broker_code
```

This overrides CCXT's defaults. CCXT will automatically:
- Call `set_ref("PASSIVBOT")` on init
- Call `approve_builder_fee("0xPASSIVBOT_ADDR", "0.02%")` on init
- Attach `builder: {"b": "0x...", "f": 20}` to every order (if approved)

**Add nag banner method** (modeled on `BinanceBot.print_new_user_suggestion`):

```python
async def print_builder_code_banner(self):
    """Print periodic banner encouraging builder code approval."""
    interval_ms = 1000 * 60 * 30  # every 30 minutes
    if hasattr(self, "_builder_banner_ts"):
        if utc_ms() - self._builder_banner_ts < interval_ms:
            return
    self._builder_banner_ts = utc_ms()

    # Skip if no builder configured or already approved
    if not isinstance(self.broker_code, dict) or "builder" not in self.broker_code:
        return
    if self.cca.options.get("approvedBuilderFee", False):
        return

    # Check approval status via info endpoint
    try:
        builder_addr = self.broker_code["builder"]
        wallet_addr = self.user_info["wallet_address"]
        res = await self.cca.fetch(
            "https://api.hyperliquid.xyz/info",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps({
                "type": "maxBuilderFee",
                "user": wallet_addr,
                "builder": builder_addr,
            }),
        )
        # If user has approved a sufficient fee, mark as approved and skip banner
        max_fee = int(res) if res else 0
        if max_fee >= self.broker_code.get("feeInt", 20):
            self.cca.options["approvedBuilderFee"] = True
            return
    except Exception as e:
        logging.debug(f"builder fee check failed: {e}")

    lines = [
        "Passivbot builder code is NOT yet approved on your Hyperliquid account.",
        " ",
        "Builder codes help fund Passivbot development at a small fee (0.02%).",
        "This is added on top of Hyperliquid's base trading fees.",
        " ",
        "To approve (one-time, requires main wallet - not agent wallet):",
        " ",
        "  Option A: Run the approval script:",
        f"    python3 tools/approve_builder_fee.py --builder {self.broker_code['builder']}",
        " ",
        "  Option B: Use the Hyperliquid Python SDK directly:",
        f'    exchange.approve_builder_fee("{self.broker_code["builder"]}", "{self.broker_code.get("feeRate", "0.02%")}")',
        " ",
        "  Option C: Approve via a third-party Hyperliquid frontend that supports builder approval.",
        " ",
        "To disable this message, set builderFee to false in your CCXT options",
        "or remove the hyperliquid entry from broker_codes.hjson.",
    ]
    front_pad = " " * 4 + "##"
    back_pad = "##"
    max_len = max(len(line) for line in lines)
    print("\n\n")
    print(front_pad + "#" * (max_len + 2) + back_pad)
    for line in lines:
        print(front_pad + " " + line + " " * (max_len - len(line) + 1) + back_pad)
    print(front_pad + "#" * (max_len + 2) + back_pad)
    print("\n\n")
```

**Override `execute_to_exchange()`** to trigger the banner (same pattern as Binance):

```python
async def execute_to_exchange(self):
    res = await super().execute_to_exchange()
    await self.print_builder_code_banner()
    return res
```

#### 3. (Optional) `tools/approve_builder_fee.py` - Standalone approval helper

A small script that users can run with their **main wallet** private key to approve
the builder fee one time. This is a convenience since the approval cannot happen
from an agent wallet.

```python
"""One-time Hyperliquid builder fee approval for Passivbot.

Usage:
    python3 tools/approve_builder_fee.py \
        --wallet-address 0xYOUR_MAIN_WALLET \
        --private-key 0xYOUR_MAIN_WALLET_PRIVATE_KEY \
        --builder 0xPASSIVBOT_BUILDER_ADDRESS \
        --fee-rate "0.02%"
"""
```

Uses CCXT directly: creates a session with the main wallet credentials and calls
`exchange.approve_builder_fee(builder, feeRate)`.

#### 4. `docs/hyperliquid_guide.md` - Add builder code section

Short section explaining:
- What builder codes are and why they exist
- One-time approval instructions (main wallet required)
- How to adjust or disable the fee
- Link to the approval helper script

### What We Do NOT Need

- **No new config keys**: `broker_codes.hjson` already handles per-exchange broker config
- **No changes to order placement code**: CCXT attaches builder info automatically via options
- **No changes to `_build_order_params()`**: Builder attachment happens inside CCXT's `create_order`
- **No manual HTTP calls for order attribution**: CCXT handles everything
- **No `live.hyperliquid_builder_enabled` config**: Unnecessary complexity

### Execution Order

1. Obtain/fund passivbot builder wallet on Hyperliquid (>= 100 USDC perps)
2. Add builder config to `broker_codes.hjson`
3. Modify `create_ccxt_sessions()` in `hyperliquid.py` to apply builder options
4. Add `print_builder_code_banner()` and `execute_to_exchange()` override
5. (Optional) Create `tools/approve_builder_fee.py` helper script
6. Update `docs/hyperliquid_guide.md`
7. Test with testnet first, then mainnet
8. Verify attribution via Hyperliquid builder fill stats:
   `https://stats-data.hyperliquid.xyz/Mainnet/builder_fills/{address}/{YYYYMMDD}.csv.lz4`

### Testing

- Verify CCXT options are correctly set in both `cca` and `ccp` sessions
- Verify `set_ref("PASSIVBOT")` is called on first API interaction
- Verify builder fee approval attempt happens on init (will fail with agent wallet - expected)
- Verify banner prints every 30 minutes when builder fee is not approved
- Verify banner stops printing once approved
- Verify orders include builder attribution after approval
- Verify bot continues to work normally when builder fee is not approved (no blocking)

### Risk Assessment

- **Low risk**: CCXT's builder code support is mature and battle-tested
- **No breaking changes**: Builder fee is purely additive; orders work with or without it
- **Silent failure**: If approval fails (agent wallet), CCXT disables builder fee and bot continues
- **User control**: Users can disable via `broker_codes.hjson` edit or CCXT options override

---

## Sources

- [Hyperliquid Builder Codes Documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/builder-codes)
- [Hyperliquid Fees Documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees)
- [Hyperliquid Builder Codes Wiki](https://hyperliquid-co.gitbook.io/wiki/guide/builder-guide/hypercore/builder-codes)
- [Hyperliquid Python SDK - Builder Fee Example](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/examples/basic_builder_fee.py)
- [CCXT Hyperliquid Documentation](https://docs.ccxt.com/exchanges/hyperliquid)
- [Freqtrade Builder Code Issue #11986](https://github.com/freqtrade/freqtrade/issues/11986) - Confirms CCXT builder fee behavior and version compatibility
- [Dwellir - Hyperliquid Builder Codes Revenue Analysis](https://www.dwellir.com/blog/hyperliquid-builder-codes)
- [Blockworks - Hyperliquid Frontend Wars](https://blockworks.co/news/hyperliquid-the-frontend-wars)
- [CCXT PR #24448 - Hyperliquid Features](https://github.com/ccxt/ccxt/pull/24448)
