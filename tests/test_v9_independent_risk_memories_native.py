"""Real-extension fixtures for independent policies over one realized-fill tape."""

from copy import deepcopy
import json

import pytest


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as native

    if getattr(native, "__is_stub__", False):
        pytest.skip("requires the compiled Rust extension")
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    return native


def payload(*, unstuck=True, gate=True, same_span=False):
    return {
        "start": 0,
        "now": 60_000,
        "known_origin": True,
        "unstuck": {"enabled": unstuck, "memory_ema_span_minutes": 3.0},
        "loss_gate": {
            "enabled": gate,
            "memory_ema_span_minutes": 3.0 if same_span else 7.0,
        },
        "fills": [
            {"identity": "loss-with-fee", "timestamp": 0, "net": -40.0},
            {"identity": "profit-with-rebate", "timestamp": 60_000, "net": 10.0},
        ],
    }


def evaluate(native, value):
    return json.loads(native.realized_memories_v9(json.dumps(value, allow_nan=False)))


@pytest.mark.parametrize("unstuck,gate", [(False, True), (True, False), (True, True), (False, False)])
def test_enabled_filters_have_independent_envelopes_and_disabled_filters_are_absent(
    require_real_passivbot_rust_module, unstuck, gate
):
    output = evaluate(require_real_passivbot_rust_module, payload(unstuck=unstuck, gate=gate))
    assert (output["start"], output["now"], output["input_fills"]) == (0, 60_000, 2)
    for name, enabled, span, drawdown in [
        ("unstuck", unstuck, 3.0, 10.0),
        ("loss_gate", gate, 7.0, 20.0),
    ]:
        if not enabled:
            assert output[name] is None
            continue
        memory = output[name]
        assert memory["span_minutes"] == span
        assert memory["drawdown"] == drawdown
        assert memory["fills"] == 2
        assert (memory["start"], memory["now"]) == (0, 60_000)
        assert memory["initialization"] == "known_zero_origin"
        assert memory["unknown_prefix"] is False
        assert memory["initial_drawdown"] == 0.0


def test_equal_policies_match_and_correction_replays_the_same_origin(
    require_real_passivbot_rust_module,
):
    native = require_real_passivbot_rust_module
    value = payload(same_span=True)
    before = evaluate(native, value)
    assert before["unstuck"] == before["loss_gate"]
    assert evaluate(native, deepcopy(value)) == before
    value["fills"][0]["net"] = -20.0
    corrected = evaluate(native, value)
    assert corrected["unstuck"]["drawdown"] == corrected["loss_gate"]["drawdown"] == 0.0
    assert corrected["start"] == before["start"] == 0


def test_unknown_prefix_remains_declared_for_each_distinct_span(
    require_real_passivbot_rust_module,
):
    value = payload()
    value.update(start=60_000, now=120_000, known_origin=False)
    value["fills"] = [{"identity": "suffix-loss", "timestamp": 60_000, "net": -40.0}]
    output = evaluate(require_real_passivbot_rust_module, value)
    for name, drawdown, attenuation in [("unstuck", 20.0, 0.5), ("loss_gate", 30.0, 0.75)]:
        memory = output[name]
        assert memory["drawdown"] == drawdown
        assert memory["unknown_prefix"] is True
        assert memory["initialization"] == "unknown_prefix_zero"
        assert memory["prefix_attenuation"] == attenuation
        assert (memory["start"], memory["now"]) == (60_000, 120_000)


def test_same_time_profits_are_not_credited_to_later_losses_before_clipping(
    require_real_passivbot_rust_module,
):
    value = payload()
    value["fills"] = [
        {"identity": "profit", "timestamp": 0, "net": 10.0},
        {"identity": "loss", "timestamp": 0, "net": -40.0},
    ]
    output = evaluate(require_real_passivbot_rust_module, value)
    assert output["unstuck"]["drawdown"] == 20.0
    assert output["loss_gate"]["drawdown"] == 30.0
    assert output["unstuck"]["same_time_order_from_canonical_manager"] is True


@pytest.mark.parametrize("defect", ["duplicate", "future", "unordered", "invalid_disabled_span"])
def test_invalid_shared_facts_and_policies_never_produce_partial_results(
    require_real_passivbot_rust_module, defect,
):
    value = payload(gate=False)
    if defect == "duplicate":
        value["fills"][1]["identity"] = value["fills"][0]["identity"]
    elif defect == "future":
        value["fills"][1]["timestamp"] = value["now"] + 1
    elif defect == "unordered":
        value["fills"].reverse()
    else:
        value["loss_gate"]["memory_ema_span_minutes"] = 0.0
    with pytest.raises(ValueError):
        evaluate(require_real_passivbot_rust_module, value)
