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

### Passivbot Builder Address

```
0x5e20A6D7e11366390Fde63EA3d0A026903359a74
```

### Approval Flow

Users must **one-time approve** a builder's maximum fee rate via the `ApproveBuilderFee` action.

**Critical constraint**: This approval **must be signed with the main wallet**, not an agent/API wallet.
Most passivbot users run with agent wallets. If the user configures their main wallet private key
in api-keys.json, passivbot can auto-approve. Otherwise, the user must approve externally.

Once approved, the bot (even with an agent wallet) can attach builder info to every order.
The user can revoke at any time.

### Querying Approval Status

```
POST https://api.hyperliquid.xyz/info
{"type": "maxBuilderFee", "user": "0xUSER", "builder": "0xBUILDER"}
```

Returns the maximum fee the user has approved for the builder. If 0 or absent, not approved.

### CCXT Built-in Support

CCXT (>= ~4.4.94) has **full built-in builder code handling** in its Hyperliquid module.
The installed passivbot version (CCXT 4.5.22) **fully supports this**.

#### How It Works in CCXT

On `initialize_client()` (called automatically on first `fetch_markets`, `create_order`, etc.):

1. **`set_ref()`** - Sets referral code via `setReferrer` action. Default: `'CCXT1'`
2. **`handle_builder_fee_approval()`** - Attempts to approve builder fee. Default builder:
   `0x6530512A6c89C7cfCEbC3BA7fcD9aDa5f30827a6` (CCXT's own address)

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
- CCXT's own default is 1 bps (`feeInt: 10`). Passivbot provides significantly more value
  than a library wrapper.
- Phantom wallet charges builder fees and earns ~$100k/day. PVP.trade has earned $7.2M lifetime.
  HeyAnon charges 10 bps (0.10%) on profitable position closes.
- Passivbot is an open-source project; 2 bps is modest and fair.
- For a typical passivbot user doing $1M monthly volume: 2 bps = ~$200/month to the project.
- At base tier with maker orders (1 bps exchange fee), a 2 bps builder fee means the user
  pays 3 bps total vs 1 bps without. Still far below taker fees (4.5 bps).
- Users can easily edit `broker_codes.hjson` to adjust if desired.

### Differences from GPT's Research

1. **GPT suggested off by default**. Our requirement is **on by default**, always-on, with
   three behavior paths depending on approval status and wallet type.

2. **GPT suggested a new config key** (`live.hyperliquid_builder_enabled`). Unnecessary -
   `broker_codes.hjson` handles this.

3. **GPT suggested explicitly disabling CCXT's default behavior**. Correct in spirit -
   we override CCXT's default builder address with passivbot's. We also suppress CCXT's
   auto-approval to control the flow ourselves.

4. **GPT missed the order-failure-as-nag pattern** for agent wallet users who haven't approved.

---

## Implementation Plan

### Design Philosophy: Builder Codes Are Always On

Builder codes are **on by default** and the bot actively nudges users toward approval.
The behavior varies by three paths:

### Three Startup Paths

#### Path A: Already Approved

At startup, query `maxBuilderFee` info endpoint.
If user has already approved passivbot's builder with sufficient fee:

- Set `approvedBuilderFee = True` on CCXT clients
- Print a **small, modest thank-you banner** at startup:
  ```
  [builder] Builder code active. Thank you for supporting Passivbot development!
  ```
- Builder info is attached to every order. No further banners.

#### Path B: Not Approved + Main Wallet Key

If not yet approved, attempt to call `cca.approve_builder_fee(builder, feeRate)`.
If it succeeds (only possible with main wallet private key):

- Set `approvedBuilderFee = True` on CCXT clients
- Print a **notice banner** explaining what happened:
  ```
      ############################################################################
      ## NOTICE: Passivbot builder code approved                               ##
      ##                                                                       ##
      ## Builder: 0x5e20A6D7e11366390Fde63EA3d0A026903359a74                   ##
      ## Fee: 0.02% (2 bps) per trade, to support Passivbot development.       ##
      ##                                                                       ##
      ## You can revoke this at any time via the Hyperliquid web interface.     ##
      ## To adjust the fee, edit broker_codes.hjson.                           ##
      ############################################################################
  ```
- Then the thank-you banner from Path A.
- Builder info is attached to every order. No further banners.

#### Path C: Not Approved + Agent Wallet Key

If approval attempt fails (agent wallets cannot sign `ApproveBuilderFee`):

1. **Set `approvedBuilderFee = True` anyway** to force CCXT to attach builder info to orders.
2. The **first order will fail** on Hyperliquid with a builder-approval error.
3. Catch this specific error in `execute_order()`:
   - Show the **nag banner** with approval instructions
   - **Temporarily disable** builder codes (`approvedBuilderFee = False`)
   - The order is not retried immediately - passivbot's normal loop will resubmit it
     next cycle, now without builder codes
4. Bot runs normally without builder attribution.
5. **Every ~30 minutes**, re-enable `approvedBuilderFee = True`:
   - First, check via `maxBuilderFee` info endpoint if user has approved in the meantime
   - If approved: permanent enable, print thank-you, stop nagging
   - If not approved: the next order will fail again, triggering another nag banner,
     then temporary disable again

This creates a "free version with ads" experience: the bot works, but the user sees
periodic reminders to approve the builder code.

### Nag Banner Content (Path C)

```
    ############################################################################################
    ## Passivbot builder code is NOT approved on your Hyperliquid account.                    ##
    ##                                                                                       ##
    ## Builder codes help fund Passivbot development at a small fee (0.02% per trade).       ##
    ##                                                                                       ##
    ## To approve (one-time), choose one of:                                                 ##
    ##                                                                                       ##
    ##   1. Switch to main wallet key in api-keys.json (bot will auto-approve on restart)    ##
    ##   2. Run: python3 src/tools/approve_builder_fee.py                                    ##
    ##   3. Approve via the Hyperliquid web interface (Settings > Approvals)                  ##
    ##                                                                                       ##
    ## To remove this message, approve the builder code.                                     ##
    ## To disable builder codes entirely, remove 'hyperliquid' from broker_codes.hjson.      ##
    ############################################################################################
```

### Early Warning During Installation

Users should be informed about builder codes **before** they hit their first bot startup,
to avoid surprise. Multiple touchpoints:

#### `api-keys.json.example` - Comment on the Hyperliquid entry

```json
"_comment_hyperliquid": "BUILDER CODES: Passivbot uses Hyperliquid builder codes (0.02% fee) to fund development. If using an agent/API wallet, approve the builder code first: python3 src/tools/approve_builder_fee.py. Or use your main wallet private_key here and the bot will auto-approve on first run.",
"hyperliquid_01" : {
    "exchange": "hyperliquid",
    "wallet_address": "wallet_address",
    "private_key": "private_key (main wallet key = auto-approve builder code; agent key = manual approval needed)",
    "is_vault": false
}
```

#### `docs/hyperliquid_guide.md` - Dedicated builder codes section

Add a prominent section after the setup instructions:

```markdown
### Builder Codes (Supporting Passivbot Development)

Passivbot includes a builder code that adds a small fee (0.02%) to each trade on Hyperliquid.
This fee goes directly to funding Passivbot development.

**If you use your main wallet private key** in api-keys.json, the bot will automatically
approve the builder code on first startup. No action needed.

**If you use an agent/API wallet** (recommended for security), you must approve the builder
code separately. Choose one method:

1. **CLI tool** (easiest):
   ```bash
   python3 src/tools/approve_builder_fee.py
   ```
   This will prompt for your main wallet private key (used once for approval, not stored).

2. **Temporary key swap**: Temporarily put your main wallet private key in api-keys.json,
   start the bot once (it will auto-approve), then switch back to your agent key.

3. **Hyperliquid web UI**: Approve via Settings > Approvals on app.hyperliquid.xyz.

The builder code can be revoked at any time. To adjust the fee or disable it entirely,
edit `broker_codes.hjson`.
```

#### `src/tools/approve_builder_fee.py` - Standalone CLI tool

Simple script that:
1. Loads builder address and fee from `broker_codes.hjson`
2. Prompts user for their main wallet address and private key
3. Creates a CCXT Hyperliquid session with main wallet credentials
4. Calls `approve_builder_fee(builder, feeRate)`
5. Confirms success

### File Changes Summary

| File | Change |
|------|--------|
| `broker_codes.hjson` | Add `hyperliquid` entry with real builder address |
| `src/exchanges/hyperliquid.py` | Builder code initialization, three-path logic, nag banner, order error handling |
| `api-keys.json.example` | Add builder code comments to Hyperliquid entry |
| `docs/hyperliquid_guide.md` | Add builder codes section |
| `src/tools/approve_builder_fee.py` | New CLI tool for one-time approval |

### Detailed Code Changes: `src/exchanges/hyperliquid.py`

#### `create_ccxt_sessions()` - Apply builder options, suppress CCXT auto-handling

```python
self._apply_builder_code_options()
```

#### `_apply_builder_code_options()` - Set CCXT options from broker_codes.hjson

- Set `ref`, `builder`, `feeRate`, `feeInt` on both `cca` and `ccp`
- Set `builderFee = False` to prevent CCXT from auto-handling (we do it ourselves)
- Set `approvedBuilderFee = False` initially

#### `_init_builder_codes()` - Async initialization, called once from first `execute_to_exchange()`

1. Query `maxBuilderFee` info endpoint
2. If approved → Path A: set `approvedBuilderFee = True`, log thank-you
3. If not approved → try `cca.approve_builder_fee(builder, feeRate)`
   - Success → Path B: log notice + thank-you, set `approvedBuilderFee = True`
   - Failure → Path C: set `approvedBuilderFee = True` (force attachment),
     set `self._builder_pending_approval = True`

#### `execute_order()` - Catch builder-fee errors (Path C)

Wrap the existing `execute_order()` to additionally catch builder-approval errors:
- Detect error strings like "Builder fee has not been approved"
- Show nag banner
- Set `approvedBuilderFee = False` on clients
- Set `self._builder_disabled_ts = utc_ms()`
- Return `{}` (empty result - order will be retried next cycle without builder codes)

#### `execute_to_exchange()` - Orchestrate init + periodic re-enable

```python
async def execute_to_exchange(self):
    # One-time builder code initialization
    if not hasattr(self, "_builder_initialized"):
        await self._init_builder_codes()
        self._builder_initialized = True

    res = await super().execute_to_exchange()

    # Periodic re-enable for Path C users
    if getattr(self, "_builder_pending_approval", False):
        await self._maybe_reenable_builder_codes()

    return res
```

#### `_maybe_reenable_builder_codes()` - Periodic re-check (every 30 min)

- If `_builder_disabled_ts` is set and >= 30 min ago:
  - Query `maxBuilderFee` to check if user approved externally
  - If approved: permanent enable, log thank-you, clear pending flag
  - If not: re-set `approvedBuilderFee = True` to force attachment on next order
    (will fail again → nag banner → temp disable → cycle continues)

### Execution Order

1. ~~Obtain/fund passivbot builder wallet~~ Done: `0x5e20A6D7e11366390Fde63EA3d0A026903359a74`
2. Update `broker_codes.hjson` with real address
3. Implement three-path logic in `hyperliquid.py`
4. Update `api-keys.json.example` with builder code comments
5. Update `docs/hyperliquid_guide.md` with builder codes section
6. Create `src/tools/approve_builder_fee.py`
7. Test all three paths (already approved, main wallet, agent wallet)
8. Verify attribution via Hyperliquid builder fill stats:
   `https://stats-data.hyperliquid.xyz/Mainnet/builder_fills/0x5e20a6d7e11366390fde63ea3d0a026903359a74/{YYYYMMDD}.csv.lz4`

### Testing

- **Path A**: Mock `maxBuilderFee` returning sufficient fee → verify thank-you log, `approvedBuilderFee = True`
- **Path B**: Mock `maxBuilderFee` returning 0, mock `approve_builder_fee` succeeding → verify notice + thank-you
- **Path C**: Mock `maxBuilderFee` returning 0, mock `approve_builder_fee` failing → verify `approvedBuilderFee = True` forced, then order error → verify nag banner + temp disable
- **Path C re-enable**: Verify 30-min timer re-enables, and that external approval is detected
- **Disable**: Remove `hyperliquid` from `broker_codes.hjson` → verify no builder behavior at all
- **All paths**: Verify bot continues trading normally, no orders permanently blocked

### Risk Assessment

- **Low risk**: Changes are additive; no existing behavior modified when builder codes are disabled
- **One order loss per nag cycle (Path C)**: First order in each 30-min cycle fails and is
  retried next loop iteration (~10-60s later). Acceptable cost for the nag mechanic.
- **User control**: Users can disable entirely by removing `hyperliquid` from `broker_codes.hjson`

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
