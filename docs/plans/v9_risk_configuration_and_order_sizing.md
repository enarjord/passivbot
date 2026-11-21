# V9 risk configuration and order sizing plan

## Release scope

Before the v9 merge, consolidate directional risk configuration, replace the separate gate and unstuck decay memories with one account-wide realized-loss memory, and change how Rust sizes and admits lossy closes. The loss gate applies across long and short. Auto-unstuck retains its own order type and eligibility rules. Panic orders bypass the gate.

This plan specifies the target behavior for the first v9 release. The release migration path is v8 → final v9. The current implementation contracts are documented in [V9 risk model](../v9_risk_model.md), [Strategy runtime](../ai/features/strategy_runtime.md), and the [Error and degradation contract](../ai/error_contract.md).

## Configuration and ownership

### Canonical schema

The example shows the affected fields with illustrative values and an enabled gate.

```json
{
  "config_version": "v9.0.0",
  "bot": {
    "portfolio": {
      "risk": {
        "hsl": {"scope": "coin"},
        "loss_realization": {
          "memory_ema_span_minutes": 43200.0,
          "loss_gate": {
            "enabled": true,
            "max_realized_drawdown_fraction": 0.01
          }
        }
      }
    },
    "global": {
      "risk": {
        "unstuck": {
          "enabled": true,
          "ema_gating_enabled": true,
          "ema_distance_fraction": -0.04,
          "ema_span_0_minutes": 450.0,
          "ema_span_1_minutes": 2400.0,
          "exposure_trigger_ratio": 0.55,
          "loss_allowance_fraction": 0.004
        }
      }
    },
    "long": {"enabled": true},
    "short": {"enabled": false}
  }
}
```

- Move directional HSL and unstuck policies to `bot.{global,long,short}.risk.hsl` and `bot.{global,long,short}.risk.unstuck`. Preserve the precedence of defaults, authored global leaves, explicit side pins, and permitted coin overrides. Side activation stays directly under each side and requires explicit activation. An entry-disabled side can still have held-position risk policy.
- Group HSL scope and the explicit portfolio HSL policy under `bot.portfolio.risk.hsl`. In `coin` and `position_side` scope, resolve local HSL policies as today. In `portfolio` scope, require an explicit portfolio policy. Preserve current HSL scope-specific override restrictions, signal arithmetic, price marks, and panic behavior.
- Define one `bot.portfolio.risk.loss_realization.memory_ema_span_minutes` and one `loss_gate` beneath that group. The span is a floating-point minute EMA span. Reuse the current finite binary32 decay validation. The default of 85,511.137 minutes preserves the starter's unstuck decay. A gate-only config that relied on the development gate's 43,200-minute default will replenish more slowly under the shared default and may request a longer bootstrap lookback. Set 43,200 explicitly to retain its decay. Keep omitted shared-span optimizer bounds fixed at the effective config value.
- The strict boolean `loss_gate.enabled` is the sole enablement authority throughout config, native planning, simulation, live admission, diagnostics and GPU evaluation. The default gate is `{"enabled": false}` with no cap. The cap is required when enabled and may be omitted when disabled. An authored cap, including a dormant one, must be finite and in `[0, 1)`. Zero is a valid active cap. Replace the old `>= 1.0` checks throughout the caller tree. Gate enablement also owns gate memory activation and bootstrap requirements.
- Retire `unstuck.close_fraction` for every config and optimizer channel. Remove the old directional top-level groups and the old portfolio loss-memory groups. Use actionable validation errors for retired paths, including dotted and nested scenario/runtime patches, source or referenced coin overrides, optimizer bounds and selectors. An old path in a partial patch must fail even if the main config uses the final layout.
- Keep unstuck's `loss_allowance_fraction` and its current applicable TWEL multiplier. The gate fraction applies directly at account scale. Neither a coin override nor a side pin sets a separate gate or decay span. The common span and gate fraction are portfolio optimizer coordinates if explicitly bounded. Gate enablement is fixed policy for a run or scenario. Validate authored fraction bounds in `[0, 1)`. Omit the fraction coordinate if all evaluated scenarios disable the gate. Where some scenarios enable it, only those scenarios consume the candidate's fraction. A numeric bound never activates the gate or supplies an omitted required authored cap.
- Change paths through the source catalog, defaults, loader and runtime projection, config writers, generated JSON Schema and field docs, examples, help, monitor, and optimizer. Migrate paths in selectors, fixed values, bounds, override allowlists, and any compose or mutation tools. Verify round trips preserve omissions, false and zero pins, and independently authored equal-valued pins. Internal parameter names may remain where they still represent the same inputs.

## Shared realized-loss memory

Use one Rust-owned drawdown state `D` in account currency. Use the existing binary32 recurrence, event sorting, per-fill clipping and validation from the v9 risk contract. Each canonical net-realized fill contributes once, including entry and exit fees, rebates, losses or gains on ordinary orders, and panic fills. Advance decay with elapsed minutes even during periods without fills. Exclude unrealized PnL and external balance flows under the existing accounting contract. HSL owns separate equity memory and episode state. A panic order's sizing is exempt from the gate, while its observed fill still affects realized-loss memory.

The state is required when either policy consumes it. An enabled gate must work with HSL and unstuck disabled. Unstuck must work with the gate disabled. When both activate, apply both formulas to the **same** raw balance and drawdown snapshot. Define:

```text
allowance(p) = max(0, p * (raw_balance + D) - D)
gate_headroom = allowance(max_realized_drawdown_fraction) when enabled
unstuck_headroom = allowance(loss_allowance_fraction * applicable_TWEL)
```

Use the current checked native FMA arithmetic. Reserve the final estimated loss against remaining batch headroom in wider precision as the gate does today. Fills replenish or consume economic headroom through the ledger. Proposed profitable orders create zero reservation and contribute zero projected credit. Carry reservations for orders that can still fill across replans until authoritative facts account for their fill or cancellation.

Consolidate history activation, bounds, warmup, coverage, restart diagnostics, and GPU accumulator state. Preserve current canonical fill replay, approximate bootstrap disclosures, recovery on repaired history, and rejection of malformed required inputs. When the gate is active and its required projection or memory is unavailable, follow the existing unavailable-input and readiness contract for affected planning. Panic follows its own readiness and authority. Remove duplicate native drawdown transport fields and update the Python and monitor consumers with the same change. Rebuild the PyO3 API, standalone unstuck helper inputs, and backtest simulation parameters around the final sizing and memory contract.

## Rust order allocation

### Close projection and common sizing

1. Begin with valid reduce-only candidates at their finalized prices and execution types. Quantities remain signed internally. Treat `qty_step`, venue minimum quantity and cost, the proposal's quantity, and remaining position quantity as explicit inputs.
2. Project **net** close PnL for the finalized route. Use the actual limit and the applicable maker or taker fee for executable versus passive behavior. Use the current market-close estimator with executable touch, taker fees and slippage for market execution. Follow the connector and backtest execution contract for crossing limits. Signed fees include rebates. Use one projection for loss classification, sizing, diagnostics and reservation. Preserve the candidate's price, route, reduce-only intent and order type during trimming.
3. A net-nonlosing close spends zero. A lossy candidate with zero remaining applicable allowance is blocked. With positive allowance and estimated net loss within it, accept the position- and venue-valid quantity. When the order's estimated loss exceeds available allowance, solve for the quantity corresponding to the allowance and round **up** to the first legal step at or above that quantity. Apply the effective venue minimum if it is larger. Round to a valid order at the relevant execution-price basis, using the existing step-aware numeric tolerances. A quantity that is exactly on a step needs zero extra steps.
4. Cap the result at the already finalized candidate quantity and the position capacity available to it. Apply the unstuck exposure bound where relevant. Revalidate final quantity, minimums, execution route, residual dust handling and projected net loss **after** any existing order normalization that can change the quantity. If the bounds contain no legal order, skip the candidate. An existing supported full-close exception for a below-minimum remaining position stays subject to its existing constraints. Unstuck requires a positive exposure target and never produces a full close. Ordinary strategy closes and other eligible reducers own any remaining dust under their existing reduce-only rules.
5. Spend the entire projected net loss of the accepted final quantity. If an upward step or venue minimum crosses the active budget, the remaining headroom for that budget becomes zero. Allow a single rounding overrun against the portfolio headroom in one planning allocation. An unstuck-specific overrun consumes the unstuck allowance and the same realized amount from the portfolio allowance. Other orders can use any remaining portfolio headroom. An exhausted budget stays exhausted during that allocation. Replanning after actual fills recomputes headroom from authoritative state.

For a $500 proposed loss with $100 available, a valid step of $100.10 is allowed. A $120 loss is allowed if the venue minimum requires it and the other quantity caps permit it. When the minimum fits inside the allowance, the batch's projected overrun is at most the loss of one quantity step. When the minimum binds above the remaining allowance, the overrun is the projected loss of the smallest valid venue order minus that remaining allowance. In the latter case the excess can be larger than the configured cap on a small account. Use the checked native headroom directly: any representable positive remainder may fund that one minimum, including a small positive rounding remainder. Apply no separate allowance tolerance or maximum overrun ratio. At zero allowance the same lossy candidates are blocked. This bound applies to projected loss per allocation. Actual fills, independent account activity, and panic may move the realized drawdown further.

### Unstuck proposals and exposure target

Preserve explicit enablement, the WE threshold, optional EMA gating, the resolved EMA spans, market/order modes, and the current close quote and order type. Preserve the current positive-unstuck-allowance eligibility requirement and one final unstuck close per account planning batch. A wider profitable-only eligibility policy is a separate decision.

Let `Q` be the absolute current quantity, `E` the average-entry price times contract multiplier, `B` the current raw sizing balance, `W` the resolved effective per-position WEL including permitted excess, and `T` the exposure trigger. Trigger when `Q * E / (B * W) > T`. At equality, the position is ineligible. The exposure bound is the greatest legal close quantity `q` satisfying:

```text
B_after = B + projected_net_close_pnl(q)
R_after = (Q - q) * E / (B_after * W)
B_after > 0
T <= R_after < R_before
```

Freeze the observed `W` during this projection. Re-resolve WEL and balance at the next plan. For fixed execution price and fee, use the linear PnL projection to solve the bound analytically or bound it in integer quantity steps. Guard ill-conditioned reciprocals and verify both the candidate lot and its upper bound against the actual projected expression. If a partial close would increase the exposure ratio because the balance falls too far, skip that unstuck proposal. Apply the same rules for shorts using signed PnL and absolute exposure.

Round the exposure bound down. Choose the largest executable quantity up to that bound that satisfies the tighter of current unstuck and available gate headroom, including the permitted loss-side overrun. When projected loss per unit is nonpositive and the candidate qualifies, the exposure bound sizes it. If the smallest legal order violates the exposure bound, skip it even if the account can afford its loss. A valid unstuck order can be smaller than the excess when the loss budget binds. Reprice and re-evaluate after market or account state changes. The floor applies to the proposed unstuck fill using the observed position. Normal strategy exits, other reducers, panic, and fills racing an order replacement may lead to lower actual exposure.

### Fillability and reducer selection

Treat panic independently. Preserve the current per-position limit of one protective reducer and its position-quantity reservation ahead of ordinary closes. Compare alternative reducers for a position at their **admissible finalized quantities**. A smaller but better-priced request can reduce more than a larger request that must be trimmed. Preserve a deterministic fallback when an alternative fails minimum sizing or eligibility. Keep eligible unstuck candidates available until one survives arbitration or all are infeasible. Re-evaluate remaining reducer alternatives against current headroom before each reservation. The reducer choice and its budget reservation commit together, so a rejected trial never consumes budget. Recheck the ordinary close capacity with the reducer's actual quantity.

Apply the following ordering to competing lossy orders within the reducer and ordinary tiers, across both sides. For reducer alternatives on the same position, select the greatest valid reduction as described above.

1. Prefer immediately executable orders using the resolved order route and current market snapshot. A crossing non-post-only sell can execute at the bid and a crossing non-post-only buy can execute at the ask. A post-only order cannot take a crossing fill. Market orders follow the existing routing policy. Respect known readiness or execution-mode restrictions.
2. If both are immediately executable, prefer the larger signed entry-distance fraction using the relevant current executable price. Long distance is `price / average_entry - 1`. Short distance is `1 - price / average_entry`. A long at −7% wins over a long at −10% if both can execute. If only the −10% candidate can execute, it wins.
3. For passive orders, prefer the smallest side-aware normalized price gap to the resting touch. A sell at the ask or buy at the bid has zero gap to its resting touch. Queue depth and future price moves can change realized fill order. Use signed entry distance for equal gaps, followed by stable identifiers such as symbol, position side, order type, price and the strategy's canonical leg identity. Resolve all meaningful ties without input iteration order. Permuting symbols or candidates must leave decisions identical by order identity, including ties.

Allocate a shared reserve in priority order. Whole proposals fitting the remaining headroom can coexist. The first oversize proposal that can be trimmed may consume the balance. Later lossy orders require positive remaining allowance. Consider all candidates available from the current decision snapshot. Candle backtests build that snapshot from current completed bars. The next candle can determine simulated execution only after candidate selection. Preserve unaffected TWEL/WEL repair diagnostics and evaluate their status from final quantities.

### Live replacement and observability

The Rust ideal set describes the target allocation and can select a replacement for a currently resting close. Live admission separately accounts for everything that can still execute. Extend the existing cancel-and-reconfirm machinery across symbols and sides when they share a reservation. Match an already-satisfied ideal to the corresponding exchange order and charge it once. If the old and replacement orders cannot coexist within the allowed budget, issue the cancel, confirm exchange state including any fills, and replan before creating the replacement. This allows a new preferred order to displace a passive one even if the passive one currently reserves the whole budget.

Resting and unresolved lossy creates that remain executable share the account budget with new creates. An observed partial fill enters the ledger and leaves only the outstanding remainder reserved. A rejection or cancellation releases the reservation once order, position and fill truth agree. Prevent concurrent order waves from spending the same headroom. Use Rust's authoritative projection for applicable admission reservations, with Python owning exchange state, reconciliation and confirmations. After planning, a route or size change that could increase projected loss requires replanning. Derive reservations from exchange and unresolved transport facts, with economic state coming from the fill ledger.

Use standard order admission while the portfolio gate is enabled and keep the Bitunix two-close post-only fast path inactive. The fast path's current admission certifies position capacity and passivity. Budget certification for that path would need to bind both creates, already resting orders, and unresolved outcomes to the same portfolio allowance. With only unstuck enabled, the ordinary wave can remain eligible because it admits only ordinary closes. Give one wave exclusive admission ownership. If an unstuck create or replacement is unresolved, use the standard path. Before issuing an unstuck create after an ordinary wave, confirm account and fill state and use fresh headroom. Apply the same explicit policy at wave selection and continuation.

Expose structured outcomes for admitted, trimmed and blocked orders with proposed and final quantity, price/route, estimated net loss, pre- and post-allocation headroom, overshoot, binding cap and reason. For unstuck, expose pre- and projected post-fill exposure and whether the order survived reducer selection. Include eligibility, budget exhaustion, and no-valid-lot cases. Keep repeated live logging bounded and route data through existing diagnostic and monitor interfaces. Update typed event payloads, reason codes and consumers together, then regenerate the [live event registry](../ai/generated/live_event_registry.md).

## Release migration

Update the [v8 → v9 tool](https://github.com/enarjord/passivbot/blob/v9/src/tools/migrate_config_v9.py) and [migration guide](../v8_to_v9_migration.md) to emit final paths directly. Preserve the existing explicit semantic acknowledgment for active v8 loss gates and retired original rolling-window policy. Choose an explicit EMA policy for active v8 gates when the old source only supplied a rolling window. Convert a v8 off sentinel to `{"enabled": false}` and omit the cap. A migrated active gate gets `enabled: true` and the matching fraction, including zero. Surface any v8 unstuck behavior that used a zero size to disable orders, and convert unambiguous disablement into the explicit flag. Report discarded sizing values and search dimensions. Require explicit choices when overrides or ranges mixed enabled and disabled semantics. Use the `v9.0.0` stamp for the first released layout. Validate structure and reject retired paths even when an old development config has that stamp.

Old optimizer checkpoints require a fresh run because the evaluator identity, search dimensions and order behavior change. Fresh evaluations may start from authored numerical seeds in the final layout. Keep each converted configuration separate from metrics computed under earlier policies.

## Implementation and acceptance

### Code ownership

- [Configuration](https://github.com/enarjord/passivbot/tree/v9/src/config) owns the authored paths, inheritance, validation, defaults, and runtime projection. [Optimizer bounds](https://github.com/enarjord/passivbot/blob/v9/src/config/optimize_bounds.py) and [v8 conversion](https://github.com/enarjord/passivbot/blob/v9/src/tools/migrate_config_v9.py) consume the same field paths.
- [Rust orchestration](https://github.com/enarjord/passivbot/blob/v9/passivbot-rust/src/orchestrator.rs) owns close finalization, arbitration and batch allowance. [Rust risk](https://github.com/enarjord/passivbot/blob/v9/passivbot-rust/src/risk.rs) owns unstuck eligibility and exposure sizing. [The native decay module](https://github.com/enarjord/passivbot/blob/v9/passivbot-rust/src/hsl_decay.rs) owns decay and allowance arithmetic.
- [Risk memory configuration](https://github.com/enarjord/passivbot/blob/v9/src/config/risk_memory.py) and [live orchestration](https://github.com/enarjord/passivbot/blob/v9/src/passivbot.py) own memory activation, acquisition horizon and planner inputs. Audit the old fraction-derived toggle in those paths and in every direct consumer.
- [Live planning inputs](https://github.com/enarjord/passivbot/blob/v9/src/live/planner_inputs.py), [reconciliation](https://github.com/enarjord/passivbot/blob/v9/src/live/reconciler.py), [execution](https://github.com/enarjord/passivbot/blob/v9/src/live/executor.py), [HSL live admission](https://github.com/enarjord/passivbot/blob/v9/src/live/hsl_live.py) and [Bitunix wave selection](https://github.com/enarjord/passivbot/blob/v9/src/live/bitunix_create_wave.py) own account facts, diagnostics projection and transport readiness. [Event emitters](https://github.com/enarjord/passivbot/blob/v9/src/live/event_emitters.py), [the event bus](https://github.com/enarjord/passivbot/blob/v9/src/live/event_bus.py), and the generated registry own the diagnostic event contract.
- [The Rust backtest](https://github.com/enarjord/passivbot/blob/v9/passivbot-rust/src/backtest.rs) owns CPU replay. [GPU parameters](https://github.com/enarjord/passivbot/blob/v9/src/optimization/gpu/parameters.py), [GPU service](https://github.com/enarjord/passivbot/blob/v9/src/optimization/gpu/service.py), [kernel dispatch](https://github.com/enarjord/passivbot/blob/v9/src/optimization/gpu/mps_kernel.py), and [the native GPU kernels](https://github.com/enarjord/passivbot/tree/v9/passivbot-rust/src/gpu) own evaluator enablement and execution layouts. Pass the boolean through each layer alongside the optional cap.

### Implementation sequence and tests

1. Define schema and native policy types, shared drawdown state, and strict validation. Update canonical source paths, generated schema, v8 conversion, and default examples in the same change series.
2. Add one checked close-loss projection and sizing helper. Integrate the portfolio allocator with the Rust orchestrator, reducer arbitration, and unstuck. Connect native diagnostics, live reservations and order replacement admission.
3. Wire the same decisions into CPU backtests and optimizer evaluation. Update Metal and CUDA screening, parameter layouts, diagnostics and replay/checkpoint versioning where supported. If a GPU route lacks the new behavior, explicitly reject or route that configuration to authoritative CPU evaluation. On native, GPU and live admission paths, verify that a disabled gate with a dormant cap below 1 has identical gate-off behavior to a disabled gate with an omitted cap. Cover memory activation, optimizer bounds, order admission, diagnostics, and Bitunix selection and continuation with standalone unstuck. Treat native Rust replay as authoritative.
4. Test config paths and inheritance through default JSON, starter, loader, authored export, CLI, coin files, fixed/scenario patches and optimizer bounds. Validate released-v8 conversion and rejection of development-v9 fields. Test shared memory against an independently replayed canonical fill sequence with mixed side fills, fees/rebates, panic, same-minute events, gaps and restarts.
5. Test long and short sizing, exact steps, $100.10 and $120 overruns, a tiny positive remainder with a high venue minimum, zero and exhausted allowance, full fits, profitable and fee-only outcomes, uneven steps, dust, venue minimum that would breach the floor, nonpositive projected balance, partial fills, cost caps, and competing reducer fallback. Assert the loss overrun cannot recur against the same depleted budget and aggregate closes fit the position. Verify the two-coin priority cases and permutations with crossed, post-only and passive quotes, including exactly tied economic priorities.
6. Rebuild the Rust extension and verify Python is using it. Exercise live-order replacement and cross-symbol budget races with the offline fake exchange. Cover restart with resting closes and input failures. Compare native CPU and supported GPU close traces including decisions, sizes and reservations, before performance qualification.
7. Replay representative EMA Anchor and Trailing Martingale configs from fresh evaluations. Isolate the gate off, old binary gate, new sized gate, and old versus new unstuck sizing. Run unstuck on and off and vary TWEL with otherwise fixed parameters. Include shared-account mixed-side, recurring entry/close and venue cost-stress cases. Compare realized-loss timing, trim size, venue-minimum overruns, time above the exposure trigger, underwater exposure, re-entry churn and equity outcomes. Publish conclusions from completed runs only. Document the change under Unreleased.
