# HSL sole-engine cutover draft

Status: incomplete implementation draft; do not merge or deploy.

This branch implements the first caller cutover from the approved
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
- Shared exchange fee and order-construction parameters have a separate module;
  the old module still contains its duplicate pending removal.

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
trace parity, live acceptance or approval of the remaining deletion. Old tests
and documentation that explicitly require legacy behavior still need migration.
The acceptance matrix in the parent plan applies to the final integrated tree.

## Known integration failures in the current draft

The broader preparation/config/override consumer suite passes 257 tests and fails
three on this draft. The corresponding preparation branch passes all 260.

- The suite selector-resolution test still exercises the removed
  `no_restart_drawdown_threshold` field; its canonical-path coverage must move to
  a supported field while retaining explicit rejection coverage for retired HSL.
- Full template-based coin override files still include retired HSL defaults.
  The template schema and generated/example configs must be migrated consistently
  before the override-foundation and relative-path tests can pass unmodified in
  purpose. Do not weaken retired-field rejection to make these fixtures load.

These failures reinforce the schema/test obligations above. The earlier focused
601-test and 367-test Rust results cover the preceding integrated caller slice,
not completion of this broader migration.
