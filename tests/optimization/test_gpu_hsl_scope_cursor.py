"""Guarded active-episode continuation versus independent fresh composition."""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import test_gpu_hsl_scope_composer as composer
from optimization.gpu.runtime import compile_shader

pytestmark = composer.pytestmark
reference = composer.reference

_BODY = r"""
    int start=info[io+1], end=info[io+2], seed=start+2;
    for(int p=0;p<count;++p) {
#if CURSOR_GUARD == 12
        pairs[p].candles=pairs[p].prices;
        pairs[p].candle_stride=1; pairs[p].candle_first=0; pairs[p].candle_last=end-1;
        pairs[p].fallback_price=100.0f;
        pairs[p].current_mark=hsl_scope_price_at(pairs[p],seed,start,seed);
#else
        pairs[p].current_mark=pairs[p].prices[(seed-start)*pairs[p].price_stride];
#endif
    }
    HslScopeCursor cursor;
    HslScopeResult result;
    bool valid=hsl_compose_scope(pairs,count,start,seed,CURSOR_GUARD==10,
        params[po+18],params[po+19],params[po+20],params[po+21],params[po+22]!=0.0f,
        result,nullptr,0,&cursor);
    if(!valid) {status[int(b)]=-3;return;}
    bool seeded=cursor.valid;
    int advanced_count=0, mismatches=0;
    for(int minute=seed+1;minute<=end;++minute) {
        HslPairSum upnl;
        hsl_pair_sum_reset(upnl,0.0f);
        for(int p=0;p<count;++p) {
            thread HslScopePair& pair=pairs[p];
            pair.current_mark=pair.prices[(minute-start)*pair.price_stride];
            float direction=pair.short_side ? -1.0f : 1.0f;
            hsl_pair_sum_add(upnl,pair.current_size==0.0f ? 0.0f
                : direction*fabs(pair.current_size)*pair.multiplier
                    *(pair.current_mark-pair.current_basis));
        }
        float budget=params[po+18], alpha=2.0f/(params[po+19]+1.0f);
        int next_start=start, next_minute=minute;
        float current_upnl=hsl_pair_sum_value(upnl), threshold=params[po+20];
#if CURSOR_GUARD == 1
        budget*=2.0f;
#elif CURSOR_GUARD == 2
        alpha*=0.5f;
#elif CURSOR_GUARD == 3
        next_start+=1;
#elif CURSOR_GUARD == 4
        next_minute-=1;
#elif CURSOR_GUARD == 5
        next_minute+=1;
#elif CURSOR_GUARD == 6
        current_upnl=NAN;
#elif CURSOR_GUARD == 7
        budget=0.0f;
#elif CURSOR_GUARD == 8
        threshold=1e-9f;
#elif CURSOR_GUARD == 9
        cursor.valid=false;
#endif
        bool advanced=hsl_advance_scope(cursor,next_start,next_minute,current_upnl,
            budget,alpha,threshold,params[po+21],params[po+22]!=0.0f,result);
        advanced_count+=advanced ? 1 : 0;
#if CURSOR_GUARD == 0
        HslScopeResult fresh;
        if(!hsl_compose_scope(pairs,count,start,minute,false,params[po+18],params[po+19],
            params[po+20],params[po+21],params[po+22]!=0.0f,fresh)) {
            status[int(b)]=-4;return;
        }
        if(!advanced || fabs(result.raw-fresh.raw)>2e-6f || fabs(result.ema-fresh.ema)>2e-6f
            || result.action!=fresh.action || result.flat_minute!=fresh.flat_minute
            || result.point_count!=fresh.point_count || result.latest_flat_minute!=fresh.latest_flat_minute
            || result.latest_flat_raw!=fresh.latest_flat_raw || result.latest_flat_ema!=fresh.latest_flat_ema)
            ++mismatches;
#else
        break;
#endif
    }
    status[int(b)]=(seeded ? 1 : 0)+2*advanced_count+1024*mismatches;
"""


@lru_cache(maxsize=16)
def _library(guard):
    source = composer._PROBE
    first = source.index("    HslScopeResult result;")
    last = source.index("    int out=int(b)*8;", first)
    source = source[:first] + _BODY + source[last:]
    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    shared = "\n".join((root / name).read_text() for name in
                       ("mps_hsl_history.metal", "mps_hsl_scope.metal"))
    return compile_shader(f"#define CURSOR_GUARD {guard}\n#include <metal_stdlib>\n"
                          "using namespace metal;\n" + shared + source)


def _row(short=False, pairs=1, constant=False):
    selected = []
    for p in range(pairs):
        direction = -1 if short and p % 2 == 0 else 1
        prices = {str(t): 100.0 if constant else 100.0 + (t % 7 - 3) * (p + 1)
                  for t in range(33)}
        selected.append(dict(symbol=f"C{p}", position=dict(
            pside="short" if direction < 0 else "long", size=direction*(p+1),
            basis=100.0, mark=prices["32"], multiplier=1.0, quantity_step=.001, inverse=False),
            fills=[dict(identity=str(p), revision=0, timestamp=1, sequence=p,
                        delta=direction*(p+1), price=100.0, realized=0.0,
                        fee=-.125*(p+1))], prices=prices))
    return dict(label="known active", pairs=selected, start=0, end=32, before=False,
                budget=1000.0, span=2.5, threshold=.99, cooldown=5, never=False, trace=False)


def _run(monkeypatch, rows, guard):
    monkeypatch.setattr(composer, "_library", lambda: _library(guard))
    return composer._run(rows)


@pytest.mark.parametrize("short,pairs", [(False, 1), (True, 1), (True, 3)])
def test_cursor_matches_fresh_at_each_mark_and_rust_endpoint(reference, monkeypatch, short, pairs):
    row = _row(short, pairs)
    _, result, statuses = _run(monkeypatch, [row], 0)
    assert statuses.tolist() == [1 + 2 * 30]
    monkeypatch.undo()
    _, expected, status = composer._run([row])
    np.testing.assert_array_equal(status, 1)
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)
    composer._check(reference, [row])


@pytest.mark.parametrize("guard", range(1, 11))
def test_cursor_rejects_changed_or_uncertain_inputs(reference, monkeypatch, guard):
    row = _row(constant=True)
    if guard == 8:
        row["pairs"][0]["fills"][0]["fee"] = 0.0
    _, _, statuses = _run(monkeypatch, [row], guard)
    assert statuses.tolist() == [0 if guard == 10 else 1]


@pytest.mark.parametrize("kind", ["missing_opening", "future_fill", "same_minute_fill", "endpoint_size", "endpoint_basis",
                                  "first_quote", "nonpositive_peak_equity"])
def test_cursor_does_not_seed_uncertain_history(reference, monkeypatch, kind):
    row = deepcopy(_row(constant=True))
    if kind == "missing_opening":
        row["pairs"][0]["fills"] = []
    elif kind == "future_fill":
        row["pairs"][0]["fills"][0]["timestamp"] = 10
    elif kind == "same_minute_fill":
        row["pairs"][0]["fills"][0]["timestamp"] = 2
    elif kind == "endpoint_size":
        row["pairs"][0]["position"]["size"] = 1.01
    elif kind == "endpoint_basis":
        row["pairs"][0]["position"]["basis"] = 100.01
    elif kind == "first_quote":
        row["pairs"][0]["prices"] = {str(t): float("nan") if t < 31 else 100.0
                                      for t in range(33)}
    else:
        row["budget"] = .01
        row["pairs"][0]["fills"][0]["fee"] = .125
        row["pairs"][0]["prices"] = {str(t): 80.0 for t in range(33)}
        row["pairs"][0]["position"]["mark"] = 80.0
    _, _, statuses = _run(monkeypatch, [row], 12 if kind == "first_quote" else 9)
    assert statuses.tolist() == [0]
