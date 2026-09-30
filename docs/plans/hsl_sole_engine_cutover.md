# HSL sole-engine cutover draft

Status: incomplete implementation draft; do not merge or deploy.

This branch implements the caller cutover and a further validated cleanup slice from the approved
[sole-engine migration plan](hsl_sole_engine_migration.md). It is not evidence that
legacy removal or integrated migration acceptance is complete.

## Implemented entrypoints

- Canonical loading selects revised when the selector is absent; explicit legacy
  requests fail with the offline migration command before runtime work.
- Python backtest preparation and native payload parsing select revised. A native
  implicit-selector regression compares fills and results with explicit revised.
- The live execution loop, account refresh lock, quote capture, reconciliation and
  connector admission use the existing revised owner. Its current-observation and
  bounded position/fill settlement contracts remain in place.
- Fake-live cycles use that same owner. Tests explicitly migrate their synthetic
  legacy fixture rather than inheriting an implicit restart-policy decision.
- Shared exchange fee and order-construction parameters have a separate module.
  Positive finite raw/sizing balance checks also have a shared owner, independent
  of legacy recovery. The old modules still contain duplicates pending removal.
- Legacy controller aliases and constructor state, the halted supervisor entrypoints,
  unreachable fill/PnL branches, fake supervisor helpers and replay benchmark are removed.
- Schema defaults and configuration validation no longer expose legacy tiers,
  terminal thresholds, grace fallback or position-during-cooldown controls.
  Newly authored examples explicitly choose `always`; canonical hydration still
  leaves missing restart choices unset and rejects later enabled scopes without a choice.
- Public unified examples explicitly author portfolio policies and portfolio optimizer
  bounds. Core HSL guides, risk guidance and rollback instructions describe the revised contract.

## Remaining implementation before migration qualification

1. Complete caller-by-caller removal of the Python legacy replay/supervisor,
   emergency journal and episode machinery. Preserve current balance validation,
   shared fee lookup, ordinary fill/PnL consumers and execution admission.
2. Remove legacy Rust controller state, simulator branches, bindings and their
   exclusive tests. Preserve revised runtime/reporting, unstuck PnL bookkeeping,
   liquidation and disabled-HSL strategy-equity analysis.
3. Complete GPU low-level dispatch and shader cleanup. Canonical revised payloads
   already use the existing revised GPU path; low-level legacy implementations and
   defaults still remain and are not qualified by this draft's Python tests.
4. Remove obsolete schema entries, metrics, replay tools and exclusive tests;
   replace the canonical feature contract and all active user instructions and
   intentional public examples. Historical changelog entries may remain.
5. Remove mechanical cutover leftovers only after their direct consumers are
   exercised. Keep this draft unmergeable while these obligations remain.

## Validation boundary

Focused configuration/native/optimizer and revised live/fake-live tests exercise
this entrypoint change. Native tests use a rebuilt source-verified extension.
This is not full-suite green, CPU/GPU performance qualification, disabled-HSL
trace parity or live acceptance. Old tests
and documentation that explicitly require legacy behavior still need migration.
The acceptance matrix in the parent plan applies to the final integrated tree.

## Current validation and remaining test migration

The earlier three configuration integration failures are resolved. The current
schema/example/migration/scenario/coin-override and retired-control suite passes
**619 tests**; documentation/CLI checks add **55**, and migrated fake assertions add **4**.
The current live/reconstruction/candle/trace/current-flat/protective/fake-live and
shared balance suites pass **786 tests**. These results cover the applied Python
cleanup; no Rust source changed in this cleanup slice. Native backtest/reporting,
CPU optimizer and GPU service/backend/CLI routing add **159 passing tests**.
These routing tests are not a new GPU throughput benchmark.

This is still not full-suite green. Legacy-only tests and legacy expectations in
mixed balance, PnL, monitor and execution suites still reference the retired
controller. Complete their retirement or migration with the corresponding revised
invariants covered before merge. The legacy history method, its helpers and the
legacy module group remain pending that coherent caller/test cleanup. Rust and
low-level GPU removal and integrated performance qualification also remain open.

## Removal boundaries and direct validation targets

| Boundary | Preserve or prove before deletion | Existing validation to extend |
| --- | --- | --- |
| `Passivbot` startup and `live/risk_input_recovery.py` | Revised startup currently returns immediately from `wait_for_startup`, but order input construction still calls `validate_balances`; retain positive finite raw/sizing balance checks in a shared owner | Revised live input/admission cases; a direct invalid-balance regression at the relocated caller |
| Legacy aliases and shared exchange parameters | Remove aliases only after checking their callers; fee and exchange parameter construction now use `live/exchange_params.py` | Revised live/fake cycles plus fee/parameter parity at order construction |
| Live owner and execution admission | Keep valid empty plans, plan receipts, protection-first service, canonical shutdown flags, queued-write checks and bounded position/fill settling | `test_hsl_revised_live.py`, `test_hsl_revised_fake_cycle.py`, `test_position_fill_sync_fake_live.py` |
| Legacy backtest state versus revised reporting | Separate obsolete controllers from strategy-equity, liquidation, unstuck and current revised reporting before removing fields or bindings | Rust tests, `test_hsl_revised_trace.py`, `test_hsl_revised_reporting.py`, disabled-HSL trace comparisons |
| GPU preparation and low-level dispatch | Reject legacy at the boundary, then remove unreachable implementations without changing revised screening/exact validation or resume semantics | GPU revised service/backend/CLI tests and the final CPU/GPU benchmark matrix |

This map is a removal checklist, not a claim that the remaining deletion has been
validated. Each implementation patch must identify its direct consumers and keep
the draft incomplete until the integrated acceptance matrix is satisfied.
