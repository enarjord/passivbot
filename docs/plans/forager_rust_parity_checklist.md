# Forager Rust Parity Checklist

This document is the implementation checklist for moving forager shortlist behavior to one centralized Rust definition while making the live path strict about required inputs.

## Target Invariants

1. Forager shortlist construction is Rust-owned trading behavior.
2. Live and backtest/orchestrator use the same Rust shortlist behavior for the same input snapshot.
3. Python only gathers data, formats config, builds payloads, and consumes results.
4. Required shortlist inputs fail loudly when missing.
5. Approved EMA fallback is only previous-EMA reuse for the same `symbol/span/path`, with age and consecutive-fallback bounds.
6. Downstream consumers assume config correctness after centralized formatting/parsing.

## File-By-File Checklist

### `AGENTS.md`

- [x] State that coin selection / forager shortlist / slot filling are Rust-owned trading behavior.
- [x] State that coin-selection inputs are trading-critical.
- [x] State that downstream config consumers must assume correctness and must not add local defaults.

### `docs/ai/principles.yaml`

- [x] Add Rust ownership of coin selection / shortlist behavior.
- [x] Add centralized config parsing/hydration contract.
- [x] Add coin-selection inputs to the strict error-handling principle.

### `docs/ai/error_contract.md`

- [x] Add coin-selection / shortlist inputs to critical-path scope.
- [x] Add downstream config-consumer local-default repair as a forbidden pattern.
- [x] Add explicit boundedness requirement for approved EMA fallback reuse.

### `docs/ai/architecture.md`

- [x] Expand Rust source-of-truth boundary to include coin selection.
- [x] Clarify that Python owns centralized config formatting/loading and payload plumbing, not shortlist behavior.

### `docs/ai/decisions.md`

- [x] Add durable decision that coin selection is Rust-owned trading behavior.
- [x] Add durable decision that config correctness is centralized.

### `src/config_utils.py`

- [ ] Keep `forager_*` config hydration/renaming/defaulting centralized here.
- [ ] Add regression tests that formatted configs always contain required shortlist keys when the feature is enabled.
- [ ] Audit touched config consumers to ensure no downstream fallback defaults remain for required shortlist params.

### `src/passivbot.py`

- [ ] Replace Python-defined shortlist behavior with Rust call-through.
- [ ] Remove `calc_forager_ema_readiness()`.
- [ ] Remove permissive required-input access in shortlist-critical code.
- [ ] Keep only orchestration/data gathering in the live shortlist path.

Proposed seam:

- `async def build_forager_candidate_payload(self, pside: str, *, max_network_fetches: Optional[int] = None) -> tuple[list[str], list[dict], dict]:`

Proposed responsibility:

1. determine candidate symbol list
2. gather strict raw feature inputs
3. build payload for Rust
4. return symbol ordering plus payload plus any non-trading-critical metadata needed by caller

Possible companion helper:

- `async def collect_forager_market_state_strict(self, pside: str, symbols: list[str], *, max_age_ms: int, max_network_fetches: Optional[int]) -> list[dict]:`

### `src/candlestick_manager.py`

- [ ] Add strict shortlist-specific batch helpers.
- [ ] Do not reuse permissive neutral-default batch helpers in shortlist-critical paths.
- [ ] Preserve permissive helpers only for noncritical/reporting callers if still needed.

Proposed strict helpers:

- `async def get_last_prices_strict(self, symbols: list[str], max_age_ms: int) -> dict[str, float]:`
- `async def get_ema_bounds_many_strict(self, items: list[tuple[str, float, float]], *, max_age_ms: Optional[int] = 60_000, timeframe: Optional[str] = None, tf: Optional[str] = None) -> dict[str, tuple[float, float]]:`
- `async def get_latest_ema_metrics_many_strict(self, requests: list[tuple[str, dict[str, float]]], *, max_age_ms: Optional[int] = None, max_network_fetches: Optional[int] = None, timeframe: Optional[str] = None, tf: Optional[str] = None) -> dict[str, dict[str, float]]:`

Alternative narrower seam if preferred:

- `async def get_forager_features_strict(...) -> dict[str, dict]:`

Rules for strict helpers:

1. complete result for all requested symbols, or raise
2. no `0.0`, `None`, `{}`, or synthetic neutral defaults for required values
3. include symbol/span/context in raised errors
4. respect one shared freshness/budget policy for the whole shortlist request

### `passivbot-rust/src/coin_selection.rs`

- [ ] Become the single Rust entry point for final shortlist ranking behavior.
- [ ] Own volume pruning, readiness calculation, weighted normalization, and deterministic tie-breaking.
- [ ] Expose a PyO3 entry point for live usage.

Proposed types:

- `pub struct ForagerCandidateInput`
- `pub struct ForagerSelectionConfig`

Proposed functions:

- `pub fn select_forager_candidates(inputs: &[ForagerCandidateInput], cfg: &ForagerSelectionConfig) -> Result<Vec<usize>, ForagerSelectionError>`
- `fn compute_forager_ema_readiness(...) -> Result<f64, ForagerSelectionError>`
- `#[pyfunction] pub fn select_forager_candidates_py(...) -> PyResult<Vec<usize>>`

Expected fields on `ForagerCandidateInput`:

1. `index`
2. `enabled`
3. `already_active`
4. `bid`
5. `ask`
6. `volume_score`
7. `volatility_score`
8. EMA inputs needed for readiness, either:
   - precomputed `ema_lower` / `ema_upper`, or
   - raw close EMA values if reuse with orchestrator is cleaner

### `passivbot-rust/src/orchestrator.rs`

- [ ] Reuse the same Rust shortlist helper instead of duplicating ranking behavior.
- [ ] Keep orchestration-only logic here: actives, forced normals, one-way blocking, per-symbol order generation.
- [ ] If possible, move readiness computation shared by orchestrator shortlist and live shortlist into a shared helper.

Preferred seam:

1. orchestrator prepares `ForagerCandidateInput`
2. orchestrator calls `select_forager_candidates(...)`
3. orchestrator consumes selected indices

### `passivbot-rust/src/python.rs`

- [ ] Parse and validate the live shortlist payload for the Rust PyO3 bridge.
- [ ] Keep validation strict and consistent with `format_config()`.
- [ ] Reject invalid required shortlist weights/config at the boundary.

Proposed additions:

- `fn extract_forager_candidate_inputs(...) -> PyResult<Vec<ForagerCandidateInput>>`
- `#[pyfunction] pub fn select_forager_candidates_py(...) -> PyResult<Vec<usize>>`

### `tests/test_format_config.py`

- [ ] Add regression tests that required `forager_*` config is present after formatting.
- [ ] Add regression tests that downstream missing-key compensation is not needed because formatting guarantees completeness.

### `tests/test_coin_filtering.py`

- [ ] Rewrite around the new live payload-builder + Rust shortlist call path.
- [ ] Add strict-failure tests for missing required shortlist features.
- [ ] Add tests proving unused features are not fetched when their weights are zero.
- [ ] Add tests proving budget exhaustion fails rather than silently degrading.

### `tests/test_orchestrator_json_api.py`

- [ ] Add parity tests showing the same shortlist/ranking behavior across live-facing Rust helper and orchestrator-facing Rust helper for the same snapshot.
- [ ] Add invalid-input tests for missing required shortlist inputs.

### `tests/test_missing_ema_fix.py`

- [ ] Add or extend tests for approved previous-EMA reuse to ensure:
  1. same `symbol/span/path`
  2. bounded by age
  3. bounded by consecutive fallback count
  4. hard-fail once the approved bound is exceeded

## Implementation Order

1. Update durable docs and decisions.
2. Add strict shortlist-specific data fetch helpers.
3. Add Rust shortlist helper and PyO3 bridge.
4. Convert live shortlist path to payload-builder + Rust call.
5. Rewire orchestrator shortlist path to the same Rust helper if needed.
6. Delete obsolete permissive shortlist code.
7. Add parity, failure, and regression tests.

## Explicit Non-Goals For This Pass

1. Repo-wide silent-error-handling sweep.
2. Refactoring unrelated permissive helpers outside the shortlist-critical path.
3. Introducing new fallback classes beyond the already-approved previous-EMA reuse.
