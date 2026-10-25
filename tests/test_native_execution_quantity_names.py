"""Current execution packet names, strict retired inputs and preserved defaults."""

from copy import deepcopy
import json

import numpy as np
import pytest

from released_v8_fixtures import released_v8_policy
from test_limit_order_fill_buffer import _native_args
from tools.migrate_config_v9 import migrate


RETIRED_NATIVE = (
    ("liquidation_threshold", "equity_floor_fraction", .05),
    ("market_order_near_touch_threshold", "market_order_near_touch_fraction", .001),
    ("market_order_slippage_pct", "market_order_slippage_fraction", .0005),
    ("limit_order_fill_buffer_pct", "limit_order_fill_buffer_fraction", 0.0),
    ("max_realized_loss_pct", "loss_gate_max_drawdown_fraction", 1.0),
)


@pytest.mark.parametrize("old,current,default", RETIRED_NATIVE)
@pytest.mark.parametrize("presence", ["retired_only", "both", "null_retired"])
def test_retired_optional_native_keys_never_fall_through_to_defaults(old, current, default, presence):
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    args = _native_args()
    packet = args[-1]
    if presence == "retired_only":
        packet.pop(current)
    packet[old] = None if presence == "null_retired" else default
    with pytest.raises(ValueError, match=f"unsupported retired native backtest field '{old}'"):
        pbr.run_backtest(*args)


def test_omitted_current_native_optional_keys_keep_exact_explicit_default_results():
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    args = _native_args()
    args[-1]["market_orders_allowed"] = True
    for _, current, default in RETIRED_NATIVE:
        args[-1][current] = default
    explicit = pbr.run_backtest(*args)
    assert len(explicit[0]) > 0
    for _, current, _ in RETIRED_NATIVE:
        args[-1].pop(current)
    omitted = pbr.run_backtest(*args)
    np.testing.assert_array_equal(explicit[0], omitted[0])
    np.testing.assert_array_equal(explicit[1], omitted[1])


def test_typed_live_global_rejects_retired_execution_fields():
    import passivbot_rust as pbr
    from test_orchestrator_json_api import make_input, make_symbol

    for old, current, default in RETIRED_NATIVE[1:3]:
        packet = make_input(balance=1000.0, symbols=[make_symbol(0, bid=100.0, ask=100.0)])
        packet["global"].pop(current, None)
        packet["global"][old] = default
        with pytest.raises(ValueError, match=old):
            pbr.compute_ideal_orders_json(json.dumps(packet))


RELEASED_FIELDS = (
    ("backtest.liquidation_threshold", "backtest.equity_floor_fraction", .07125),
    ("live.market_order_near_touch_threshold", "bot.portfolio.execution.market_order_near_touch_fraction", .000875),
    ("backtest.market_order_slippage_pct", "backtest.market_order_slippage_fraction", .00125),
    ("backtest.limit_order_fill_buffer_pct", "backtest.limit_order_fill_buffer_fraction", .000375),
)


@pytest.mark.parametrize("owner", ["root", "fixed", "scenario"])
def test_released_v8_execution_fields_keep_values_final_paths_and_sources(owner):
    source = released_v8_policy()
    values = {old: value for old, _, value in RELEASED_FIELDS}
    if owner == "root":
        for old, value in values.items():
            section, leaf = old.split(".")
            source[section][leaf] = value
    elif owner == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = values
    else:
        source["backtest"]["scenarios"] = [{"label": "execution", "overrides": values}]
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert source == original
    converted = result.config
    patch = (converted["optimize"]["fixed_runtime_overrides"] if owner == "fixed" else
             converted["backtest"]["scenarios"][0]["overrides"] if owner == "scenario" else None)
    for old, current, value in RELEASED_FIELDS:
        if patch is not None:
            assert patch[current] == value
        else:
            node = converted
            for part in current.split("."):
                node = node[part]
            assert node == value
        assert any(old in change["path"] for change in result.changes)
