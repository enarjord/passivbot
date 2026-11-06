"""Unavailable trailing projections remain unknown through their presentation consumers."""

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from live.event_emitters import (
    _trailing_status_summary,
    _trailing_threshold_projection,
    emit_live_event,
    emit_trailing_status_event,
)
from live.event_bus import ListEventSink, LiveEventPipeline
from test_trailing_diagnostic_projections import _native_case, DEFAULT_EXTREMA, FINITE_EXTREMA
from test_trailing_diagnostics import _hype_trailing_martingale_close_inputs
from trailing_diagnostics import build_trailing_martingale_close_diagnostic


def _emitted_data(payload, side):
    sink = ListEventSink()

    class Bot:
        _emit_live_event = emit_live_event

        def __init__(self):
            self.exchange = "fake"
            self.user = "fake_test"
            self.bot_id = "fake_test"
            self._live_event_pipeline = LiveEventPipeline(structured_sinks=[sink])

        def _current_live_event_cycle_id(self):
            return "test_cycle"

    bot = Bot()
    try:
        emit_trailing_status_event(
            bot, symbol="BTC/USDT:USDT", pside=side,
            kind=payload["kind"], payload=payload, changed=True, operator_visible=True,
        )
        assert bot._live_event_pipeline.flush(timeout=2.0)
        return sink.events[-1].data
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0)


@pytest.mark.parametrize("builder_kind", ["entry", "close", "martingale_close"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_native_reset_projection_preserves_unknown_comparisons_through_event(
    monkeypatch, builder_kind, side
):
    payload = _native_case(monkeypatch, builder_kind, side, DEFAULT_EXTREMA)
    original = deepcopy(payload)

    data = _emitted_data(payload, side)

    assert data["retracement_met"] is None
    assert data["retracement_met_status"] == "unavailable"
    assert data["retracement_met_reason"] == "trailing_extremum_uninitialized"
    assert data["retracement_price"] is None
    assert data["retracement_price_status"] == "unavailable"
    assert data["projection_status"] == "unavailable"
    assert data["trailing_status"] == payload["status"]
    json.dumps(data, allow_nan=False)
    assert payload == original


@pytest.mark.parametrize("value", [True, False, None])
def test_event_summary_preserves_three_state_comparison(value):
    summary = _trailing_status_summary({"threshold_met": value, "retracement_met": value})

    assert summary["threshold_met"] is value
    assert summary["retracement_met"] is value


def test_event_summary_retains_unavailable_native_fraction_classification():
    payload = {
        "threshold_fraction": None,
        "threshold_fraction_status": "unavailable",
        "threshold_fraction_reason": "projection_fraction_unavailable",
        "threshold_met": None,
        "threshold_met_status": "unavailable",
        "threshold_met_reason": "projection_fraction_unavailable",
        "projection_status": "unavailable",
    }

    summary = _trailing_status_summary(payload)

    assert all(summary[key] == value for key, value in payload.items())
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("field", ["threshold", "retracement"])
def test_native_fraction_overflow_reason_survives_structured_event(field):
    inputs = _hype_trailing_martingale_close_inputs()
    inputs["trailing"] = deepcopy(FINITE_EXTREMA)
    inputs["close_params"][f"{field}_base_fraction"] = 1e308
    inputs["close_params"][f"{field}_log_range_1m_coefficient"] = 1e308
    inputs["state"]["log_range_ema_1m"] = 1.0
    payload = build_trailing_martingale_close_diagnostic(inputs)

    data = _emitted_data(payload, "long")

    for suffix in ("fraction", "price", "met"):
        key = f"{field}_{suffix}"
        assert data[key] is None
        assert data[f"{key}_status"] == "unavailable"
        assert data[f"{key}_reason"] == "projection_fraction_unavailable"
    json.dumps(data, allow_nan=False)


@pytest.mark.parametrize("kind,side", [("entry", "long"), ("close", "short")])
def test_event_summary_does_not_reintroduce_infinite_auxiliary_projection(kind, side):
    assert _trailing_threshold_projection(
        kind=kind, pside=side, threshold_price=1e308, retracement_fraction=1.0
    ) is None
    summary = _trailing_status_summary({
        "kind": kind, "pside": side,
        "threshold_price": 1e308, "retracement_fraction": 1.0,
    })

    assert summary.get("threshold_projection_retracement_price") is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("kind,side", [("entry", "short"), ("close", "long")])
def test_event_projection_preserves_genuine_zero(kind, side):
    assert _trailing_threshold_projection(
        kind=kind, pside=side, threshold_price=100.0, retracement_fraction=1.0
    ) == 0.0


@pytest.mark.skipif(shutil.which("node") is None, reason="dashboard smoke requires Node.js")
def test_dashboard_renders_null_comparisons_as_unknown_without_hiding_false():
    # Exercise the actual renderer with only its DOM/presentation dependencies stubbed.
    source = Path("src/monitor_dashboard_static/dashboard.js").read_text()
    functions = []
    for name in ("trailingComparisonLabel", "renderTrailing"):
        match = re.search(rf"^  function {name}\(.*?^  }}", source, re.MULTILINE | re.DOTALL)
        assert match is not None
        functions.append(match.group())
    script = "\n".join(functions) + """
const assert = require('node:assert/strict');
const rows = [null, undefined, false, true].map(value => ({
  symbol: 'BTC/USDT:USDT', pside: 'long', kind: 'entry',
  payload: {threshold_met: value, retracement_met: value, status: 'unavailable'}
}));
const items = [];
const els = {trailingList: {innerHTML: '', appendChild: node => items.push(node)}};
const document = {createElement: () => ({dataset: {}})};
const filteredTrailingRows = () => rows;
const focusClasses = () => '';
const escapeHtml = value => String(value);
const shortSymbol = value => value;
const trailingStatusTone = () => '';
const fmtCompact = value => value == null ? '-' : String(value);
renderTrailing({}, null);
assert.equal(items.length, 4);
for (const [index, label] of ['unknown', 'unknown', 'wait', 'met'].entries()) {
  assert.ok(items[index].innerHTML.includes(`thr ${label} · ret ${label}`));
}
"""
    result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
