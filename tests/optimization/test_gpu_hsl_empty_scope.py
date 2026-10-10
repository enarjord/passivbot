"""Native empty retained histories reduce to the same actual-position singleton."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import test_gpu_hsl_scope_composer as composer
from optimization.gpu.runtime import compile_shader

pytestmark = composer.pytestmark
reference = composer.reference

_BODY = r"""
    HslScopeResult result, fresh;
    bool used=hsl_compose_empty_native_scope(pairs,count,info[io+1],info[io+2],
        params[po+18],params[po+19],params[po+20],params[po+21],params[po+22]!=0.0f,result);
    bool valid=hsl_compose_scope(pairs,count,info[io+1],info[io+2],false,
        params[po+18],params[po+19],params[po+20],params[po+21],params[po+22]!=0.0f,
        fresh,points+int(b)*512,512);
    if(!valid) {status[int(b)]=-1;return;}
    if(used && (result.raw!=fresh.raw || result.ema!=fresh.ema || result.action!=fresh.action
        || result.flat_minute!=fresh.flat_minute || result.point_count!=fresh.point_count
        || result.latest_flat_minute!=fresh.latest_flat_minute
        || result.latest_flat_raw!=fresh.latest_flat_raw || result.latest_flat_ema!=fresh.latest_flat_ema))
        {status[int(b)]=-2;return;}
    if(!used) result=fresh;
    status[int(b)]=used ? 1 : 2;
"""


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    source = "\n".join((root / name).read_text() for name in
                       ("mps_hsl_history.metal", "mps_hsl_scope.metal"))
    probe = composer._PROBE.replace("device const float* prices", "constant float* prices")
    probe = probe.replace("pair.prices=prices+(int(b)*3+p)*160;", "pair.prices=nullptr;")
    probe = probe.replace("pair.candles=nullptr; pair.candle_stride=0; pair.fallback_price=0.0f;",
                          "pair.candles=prices+(int(b)*3+p)*160; pair.candle_stride=1; "
                          "pair.candle_first=0; pair.candle_last=info[io+2]-1; pair.fallback_price=0.0f;")
    first = probe.index("    HslScopeResult result;")
    last = probe.index("    int out=int(b)*8;", first)
    probe = probe[:first] + _BODY + probe[last:]
    return compile_shader("#include <metal_stdlib>\nusing namespace metal;\n" + source + probe)


def _rows(span, never):
    rows = []
    for count in (1, 2, 3):
        for change in (-20.0, 0.0, 20.0):
            pairs = []
            for p in range(count):
                sign = -1 if p == 1 else 1
                pairs.append(dict(symbol=str(p), position=dict(
                    size=sign * (p + 1), basis=100.0, mark=100.0 + sign * change,
                    multiplier=1.0 + p*.5, quantity_step=1.0, inverse=False,
                    pside="short" if sign < 0 else "long"), fills=[],
                    # Missing prefix is projected from the first causal quote.
                    prices={str(t):0.0 if t < 3 else 100.0 for t in range(35)}))
            rows.append(dict(label=f"empty,count={count},change={change}", pairs=pairs,
                             start=0, end=34, before=False, budget=500.0, span=span,
                             threshold=.1, cooldown=2, never=never))
    return rows


@pytest.mark.parametrize("span", [1.0, 2.5, 25.0])
@pytest.mark.parametrize("never", [False, True])
def test_empty_scope_is_identical_to_fresh_and_matches_rust(reference, monkeypatch, span, never):
    monkeypatch.setattr(composer, "_library", _library)
    composer._check(reference, _rows(span, never))


def test_retained_execution_and_unusable_quotes_decline_shortcut(monkeypatch):
    monkeypatch.setattr(composer, "_library", _library)
    row = composer._cases(False, 2.5, True)[0]
    _, _, status = composer._run([row])
    np.testing.assert_array_equal(status, 2)
    row = _rows(2.5, True)[0]
    row["pairs"][0]["prices"] = {str(t):0.0 for t in range(35)}
    _, _, status = composer._run([row])
    np.testing.assert_array_equal(status, -1)
