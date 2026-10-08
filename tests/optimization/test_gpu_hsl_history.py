"""GPU factual reconstruction agrees with Rust before controller composition."""

from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optimization.gpu.runtime import compile_shader, gpu_device

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="GPU required"
)

FACT_CAPACITY = 128
PRICE_CAPACITY = 160


@pytest.fixture(scope="module")
def reference():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()
    return passivbot_rust


_PROBE = r"""
#define FACT_CAPACITY 128
#define PRICE_CAPACITY 160
kernel void history_probe(
    device const float* params, device const int* info, device const float* fills,
    device const int* fill_times, device const float* prices, device const int* price_times,
    device float* events, device float* output, device float* opening, device int* status,
    uint b [[thread_position_in_grid]]
) {
    int po = int(b) * 6, io = int(b) * 5;
    int fbase = int(b) * FACT_CAPACITY * 4, tbase = int(b) * FACT_CAPACITY;
    int sbase = int(b) * PRICE_CAPACITY, obase = sbase * 4;
    HslPairFacts facts;
    facts.values = (device const HslPairFact*)(fills + fbase);
    facts.minutes = fill_times + tbase;
    facts.head = info[io + 4]; facts.count = info[io]; facts.capacity = FACT_CAPACITY;
    facts.value_stride = facts.minute_stride = 1;
    device HslPairEvent* result = (device HslPairEvent*)(events + fbase);
    HslPairHistory history;
    bool short_side = params[po + 4] < 0.0f;
    bool valid = hsl_reconstruct_pair(facts, params[po], params[po+1], short_side,
        params[po+5], result, history);
    status[int(b)] = valid ? 1 : -1;
    if (!valid) return;
    opening[int(b)*2] = history.opening_size;
    opening[int(b)*2+1] = history.opening_basis;
    int consumed = 0, end = info[io+3]; bool before = info[io+2] != 0;
    for (int i=0; i<info[io+1]; ++i) {
        int minute = price_times[sbase+i];
        while (consumed < facts.count && hsl_pair_fill_precedes_price(
            hsl_pair_minute(facts,consumed), minute,end,before)) ++consumed;
        HslPairSample sample;
        if (!hsl_sample_pair(history,result,consumed,facts.count,minute,end,before,
            params[po],params[po+1],params[po+2],prices[sbase+i],params[po+3],
            short_side,sample)) {
            status[int(b)] = -2; return;
        }
        int at = obase + i*4;
        output[at]=sample.size; output[at+1]=sample.basis;
        output[at+2]=sample.realized; output[at+3]=sample.upnl;
    }
}
"""


@lru_cache(maxsize=1)
def _library():
    source = (Path(__file__).resolve().parents[2] / "passivbot-rust" / "src" / "gpu"
              / "mps_hsl_history.metal").read_text()
    return compile_shader("#include <metal_stdlib>\nusing namespace metal;\n" + source + _PROBE)


def _run(rows):
    n = len(rows)
    params = np.zeros((n, 6), np.float32)
    info = np.zeros((n, 5), np.int32)
    facts = np.zeros((n, FACT_CAPACITY, 4), np.float32)
    times = np.zeros((n, FACT_CAPACITY), np.int32)
    prices = np.zeros((n, PRICE_CAPACITY), np.float32)
    price_times = np.zeros((n, PRICE_CAPACITY), np.int32)
    for b, row in enumerate(rows):
        p, f = row["position"], row["fills"]
        head = row["head"]
        params[b] = [p["size"], p["basis"], p["mark"], p["multiplier"],
                     -1 if p["pside"] == "short" else 1, p["quantity_step"]]
        info[b] = [row.get("fact_count", len(f)), len(row["prices"]), row["fills_before_same_time_price"],
                   row["end"], head]
        for i, fill in enumerate(f):
            slot = (head + i) % FACT_CAPACITY
            facts[b, slot] = [fill[k] for k in ("delta", "price", "realized", "fee")]
            times[b, slot] = fill["timestamp"]
        prices[b, :len(row["prices"])] = list(row["prices"].values())
        price_times[b, :len(row["prices"])] = [int(t) for t in row["prices"]]
    device = gpu_device()
    inputs = [torch.from_numpy(a).to(device) for a in
              (params, info, facts, times, prices, price_times)]
    outputs = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in (
        ((n, FACT_CAPACITY, 4), torch.float32), ((n, PRICE_CAPACITY, 4), torch.float32),
        ((n, 2), torch.float32), ((n,), torch.int32))]
    _library().history_probe(*inputs, *outputs, threads=n)
    return tuple(v.cpu().numpy() for v in outputs)


def _row(label, tape, quantity, basis, step, direction, before, *, clipped=0, head=0):
    end = 1 + len(tape) // 2 + 3
    return dict(
        label=label, head=head, start=0 if clipped == 0 else 1 + clipped // 2, end=end,
        fills_before_same_time_price=before,
        position=dict(size=direction * quantity, basis=basis, mark=85., multiplier=1.,
                      quantity_step=step, inverse=False, pside="long" if direction == 1 else "short"),
        fills=[dict(identity=str(i), timestamp=1+(i+clipped)//2, delta=direction*q,
                    price=p, realized=direction*g, fee=-.125, sequence=i, revision=0)
               for i, (q, p, g) in enumerate(tape[clipped:])],
        prices={str(t):90.+(t*7)%23 for t in range(0 if clipped == 0 else 1+clipped//2, end+1)},
    )


def _cases(direction, before):
    rng = random.Random(7121)
    rows = []
    for seed in range(240):
        step = (1e-6, .001, .1, 1., 100.)[seed % 5]
        size, basis, tape = 0, 0., []
        for _ in range(64):
            add = size == 0 or rng.random() < .57
            q = rng.randrange(1, 1001) if add else -rng.randrange(1, size+1)
            price = rng.randrange(8000, 12001)/100
            gross = 0. if q > 0 else -q*step*(price-basis)
            if q > 0:
                basis = (size*basis+q*price)/(size+q)
            size += q
            if not size:
                basis = 0.
            tape.append((q*step, price, gross))
        rows.append(_row(seed, tape, size*step, basis, step, direction, before,
                         clipped=seed % 60, head=seed % FACT_CAPACITY))
    authored = [
        ("empty_flat", [], 0., 0., .1), ("empty_held", [], .3, 100., .1),
        ("missing_close", [(.3, 100., 0.)], 0., 0., .1),
        ("close_only", [(-.1, 90., -1.), (-.2, 95., -1.)], .1, 100., .1),
        ("interior_deficit", [(.1, 100., 0.), (-.3, 90., -3.), (.2, 80., 0.)], .2, 80., .1),
        ("full_close", [(.1, 100., 0.), (.2, 100., 0.), (-.3, 90., -3.)], 0., 0., .1),
        ("one_tick_tail", [(262144., 100., 0.), (-262143., 90., -2621430.)], 1., 100., 1.),
        ("cash_cancellation", [(1., 100., 16777216.), (-1., 90., -16777216.), (1., 80., 0.)], 1., 80., 1.),
        ("same_time_reopen", [(.3, 100., 0.), (-.3, 90., -3.), (.1, 80., 0.)], .1, 80., .1),
    ]
    rows.extend(_row(*case, direction, before, head=127) for case in authored)
    return rows


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("before", [False, True])
def test_retained_fractional_history(reference, direction, before):
    rows = _cases(direction, before)
    events, samples, openings, statuses = _run(rows)
    np.testing.assert_array_equal(statuses, 1)
    eps = np.finfo(np.float32).eps
    for b, row in enumerate(rows):
        payload = {k:v for k, v in row.items() if k not in ("label", "head")}
        ref = json.loads(reference.hsl_history(json.dumps(payload)))
        p, f = row["position"], row["fills"]
        price_scale = max([p["basis"], p["mark"], *row["prices"].values(),
                           *[v["price"] for v in f]])
        size_scale = max([abs(p["size"]), *[abs(v["delta"]) for v in f], 1e-30])*max(1, len(f))
        cash_scale = sum(abs(v["realized"])+abs(v["fee"]) for v in f)
        # Near-zero cashflow/UPNL is conditioned by contributing magnitudes.
        # These component guards do not change native metric tolerances.
        qty_bound = min(p["quantity_step"]*.25, 8*eps*size_scale)+1e-12
        basis_bound, cash_bound = 32*eps*price_scale, 8*eps*cash_scale+1e-7
        want_events = np.asarray([[v[k] for k in ("before", "after", "basis", "realized_cumsum")]
                                  for v in ref["events"]]).reshape(-1, 4)
        actual = events[b, :len(want_events)]
        assert np.all(np.abs(actual-want_events) <= [qty_bound, qty_bound, basis_bound, cash_bound]), row["label"]
        np.testing.assert_array_equal(actual[:, 1] == 0., want_events[:, 1] == 0., err_msg=str(row["label"]))
        want = np.asarray([[v[k] for k in ("size", "basis", "pnl", "upnl")] for v in ref["samples"]])
        actual = samples[b, :len(want)]
        sample_prices = np.asarray([p["mark"] if v["timestamp"] == row["end"]
                                    else row["prices"][str(v["timestamp"])] for v in ref["samples"]])
        upnl_bound = (np.abs(want[:, 0])*(basis_bound+eps*price_scale)
                      + qty_bound*(np.abs(sample_prices-want[:, 1])+basis_bound)
                      + 4*eps*np.abs(want[:, 3])+1e-7)
        bounds = np.column_stack([np.full(len(want), v) for v in
                                  (qty_bound, basis_bound, cash_bound)] + [upnl_bound])
        assert np.all(np.abs(actual-want) <= bounds), row["label"]
        np.testing.assert_array_equal(actual[:, 0] == 0., want[:, 0] == 0., err_msg=str(row["label"]))
        assert np.all(np.abs(openings[b]-[ref["opening_size"], ref["opening_basis"]])
                      <= [qty_bound, basis_bound]), row["label"]
        if row["label"] == "cash_cancellation":
            np.testing.assert_array_equal(actual[:, 2], want[:, 2])
        if row["label"] == "one_tick_tail":
            assert actual[-1, 0] == direction


@pytest.mark.parametrize("field,value", [
    ("size", float("nan")), ("size", -1.), ("basis", float("nan")), ("basis", 0.),
    ("quantity_step", 0.), ("quantity_step", float("nan")), ("mark", 0.), ("multiplier", 0.),
    ("delta", 0.), ("delta", float("nan")), ("price", 0.), ("price", float("inf")),
    ("realized", float("nan")), ("fee", float("inf")), ("head", -1), ("head", FACT_CAPACITY),
    ("fact_count", -1), ("fact_count", FACT_CAPACITY+1),
])
def test_invalid_factual_inputs(reference, field, value):
    row = _row("invalid", [(1., 100., 0.)], 1., 100., 1., 1, True)
    if field in row["position"]:
        row["position"][field] = value
    elif field in ("head", "fact_count"):
        row[field] = value
    else:
        row["fills"][0][field] = value
    assert _run([row])[-1][0] < 0


_ENDPOINT_PROBE = r"""
kernel void endpoint_probe(
    device const float* input, device int* state, device HslPairRecord* records,
    device float* output, uint b [[thread_position_in_grid]]
) {
    HslPairRing ring;
    ring.state=reinterpret_cast<device HslPairRingState*>(state);
    ring.records=records; ring.capacity=4; ring.lookback=100; ring.enabled=true;
    hsl_pair_ring_reset(ring);
    output[0]=ring.state->current_size; output[1]=ring.state->current_basis;
    for(int i=0;i<4;++i) {
        HslPairFact fact;
        fact.delta=input[i*7]; fact.price=input[i*7+1];
        fact.realized=0.0f; fact.fee=-.125f;
        bool accepted=hsl_pair_ring_capture(ring,fact,1,i,input[i*7+2],input[i*7+3],
            input[i*7+4]!=0.0f);
        output[2+i*4]=float(accepted); output[3+i*4]=float(ring.state->failure);
        output[4+i*4]=ring.state->current_size; output[5+i*4]=ring.state->current_basis;
    }
}
"""


@lru_cache(maxsize=1)
def _endpoint_library():
    source = (Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu/mps_hsl_history.metal").read_text()
    return compile_shader("#include <metal_stdlib>\nusing namespace metal;\n" + source + _ENDPOINT_PROBE)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("invalid", [None, "held_zero", "held_nan", "held_inf", "held_negative", "flat_nonzero"])
def test_native_endpoint_capture_keeps_actual_basis_and_rejects_malformed_inputs(reference, direction, invalid):
    facts = np.array([[direction*.1,100.,direction*.1,100.,direction<0,0,0],
                      [direction*.1,120.,direction*.2,110.,direction<0,0,0],
                      [-direction*.1,110.,direction*.1,110.,direction<0,0,0],
                      [-direction*.1,110.,0.,0.,direction<0,0,0]], np.float32)
    invalid_at = 3 if invalid == "flat_nonzero" else 2
    if invalid is not None:
        facts[invalid_at, 3] = dict(held_zero=0., held_nan=float("nan"), held_inf=float("inf"),
                                    held_negative=-1., flat_nonzero=110.)[invalid]
    device = gpu_device()
    inputs = torch.from_numpy(facts).to(device)
    state = torch.empty(16, dtype=torch.int32, device=device)
    records = torch.empty((4, 8), device=device)
    output = torch.empty(18, device=device)
    _endpoint_library().endpoint_probe(inputs, state, records, output, threads=1)
    actual = output.cpu().numpy()
    np.testing.assert_array_equal(actual[:2], 0.)
    rows = actual[2:].reshape(4, 4)
    for i, row in enumerate(rows):
        rejected = invalid is not None and i >= invalid_at
        np.testing.assert_array_equal(row[:2], [0, 2] if rejected else [1, 0])
        source = facts[invalid_at-1] if rejected else facts[i]
        np.testing.assert_array_equal(row[2:], source[2:4])

_RING_PROBE = r"""
#define RING_PROBE_STEPS 16
kernel void fact_ring_probe(
    device const float* facts, device const int* options,
    device const int* chronology, device int* state_words,
    device HslPairRecord* records, device float* output,
    uint b [[thread_position_in_grid]]
) {
    int io = int(b)*4, base = int(b)*RING_PROBE_STEPS;
    HslPairRing ring;
    ring.state = reinterpret_cast<device HslPairRingState*>(state_words + int(b)*16);
    ring.records = records + base;
    ring.capacity = options[io]; ring.lookback = options[io+1];
    ring.enabled = options[io+3] != 0;
    hsl_pair_ring_reset(ring);
    for (int i=0; i<options[io+2]; ++i) {
        int at = (base+i)*6, ct = (base+i)*3, out = (base+i)*15;
        if (chronology[ct+2] != 0) hsl_pair_ring_reset(ring);
        HslPairFact fact;
        fact.delta=facts[at]; fact.price=facts[at+1];
        fact.realized=facts[at+2]; fact.fee=facts[at+3];
        bool accepted = hsl_pair_ring_append(ring,fact,chronology[ct],
            chronology[ct+1],facts[at+4],facts[at+5]<0);
        for (int j=0; j<15; ++j) output[out+j]=0.0f;
        output[out]=accepted; output[out+1]=ring.state->head;
        output[out+2]=ring.state->count; output[out+3]=ring.state->version;
        output[out+4]=ring.state->failure;
        output[out+13]=float(sizeof(HslPairRingState));
        output[out+14]=float(sizeof(HslPairRecord));
        if (ring.state->count > 0) {
            int last=(ring.state->head+ring.state->count-1)%ring.capacity;
            HslPairRecord record=ring.records[last];
            output[out+5]=record.fact.delta; output[out+6]=record.fact.price;
            output[out+7]=record.fact.realized; output[out+8]=record.fact.fee;
            output[out+9]=record.actual_size_after;
            output[out+10]=record.first_sequence; output[out+11]=record.last_sequence;
            output[out+12]=record.minute;
        }
    }
}
kernel void ring_history_probe(
    device int* state_words, device HslPairRecord* records,
    device const float* params, device const int* options,
    device HslPairEvent* events, device float* samples, device float* opening,
    uint b [[thread_position_in_grid]]
) {
    int po=int(b)*5, io=int(b)*4;
    HslPairRing ring;
    ring.state=reinterpret_cast<device HslPairRingState*>(state_words+int(b)*16);
    ring.records=records+int(b)*RING_PROBE_STEPS;
    ring.capacity=options[io]; ring.lookback=0; ring.enabled=true;
    HslPairFacts facts=hsl_pair_ring_view(ring,options[io+1]);
    HslPairHistory history;
    bool short_side=params[po+4]<0;
    if (!hsl_reconstruct_pair(facts,params[po],params[po+1],short_side,
        params[po+3],events+int(b)*RING_PROBE_STEPS,history)) return;
    opening[int(b)*2]=history.opening_size;
    opening[int(b)*2+1]=history.opening_basis;
    int consumed=0;
    for (int minute=options[io+1]; minute<=options[io+2]; ++minute) {
        while (consumed<facts.count && hsl_pair_fill_precedes_price(
            hsl_pair_minute(facts,consumed),minute,options[io+2],options[io+3]!=0)) ++consumed;
        HslPairSample sample;
        if (!hsl_sample_pair(history,events+int(b)*RING_PROBE_STEPS,consumed,facts.count,
            minute,options[io+2],options[io+3]!=0,params[po],params[po+1],params[po+2],
            95.0f,1.0f,short_side,sample)) return;
        int out=(int(b)*RING_PROBE_STEPS+minute-options[io+1])*4;
        samples[out]=sample.size; samples[out+1]=sample.basis;
        samples[out+2]=sample.realized; samples[out+3]=sample.upnl;
    }
}
"""


@lru_cache(maxsize=1)
def _ring_library():
    source = (Path(__file__).resolve().parents[2] / "passivbot-rust" / "src" / "gpu"
              / "mps_hsl_history.metal").read_text()
    return compile_shader("#include <metal_stdlib>\nusing namespace metal;\n" + source + _RING_PROBE)


def _ring_run(tape, *, capacity=8, lookback=10, direction=1, enabled=True, buffers=False):
    facts = np.zeros((1, 16, 6), np.float32)
    chronology = np.zeros((1, 16, 3), np.int32)
    for i, (qty, price, gross, fee, after, minute, sequence, reset) in enumerate(tape):
        facts[0, i] = [direction*qty, price, direction*gross, fee, direction*after, direction]
        chronology[0, i] = [minute, sequence, reset]
    options = np.array([[capacity, lookback, len(tape), enabled]], np.int32)
    device = gpu_device()
    inputs = [torch.from_numpy(v).to(device) for v in (facts, options, chronology)]
    states = torch.empty((1, 16), dtype=torch.int32, device=device)
    records = torch.empty((1, 16, 8), dtype=torch.float32, device=device)
    out = torch.empty((1, 16, 15), dtype=torch.float32, device=device)
    _ring_library().fact_ring_probe(*inputs, states, records, out, threads=1)
    result = out.cpu().numpy()[0, :len(tape)]
    assert np.all(result[:, 13] <= 64)
    np.testing.assert_array_equal(result[:, 14], 32)
    return (result, states, records) if buffers else result


@pytest.mark.parametrize("direction", [1, -1])
def test_ring_coalesces_only_consecutive_runs_and_keeps_flat_reopen(reference, direction):
    tape = [
        (.1, 100., 0., -.125, .1, 1, 0, False),
        (.2, 130., 0., -.125, .3, 1, 1, False),
        (-.1, 90., -1., -.125, .2, 1, 2, False),
        (-.2, 80., -4., -.125, 0., 1, 3, False),
        (.1, 70., 0., -.125, .1, 1, 4, False),
        (.1, 80., 0., -.125, .2, 1, 6, False),  # Another pair filled at sequence 5.
    ]
    out = _ring_run(tape, direction=direction)
    np.testing.assert_array_equal(out[:, 0], 1)
    np.testing.assert_array_equal(out[:, 2], [1, 1, 2, 2, 3, 4])
    np.testing.assert_allclose(out[1, 5:10], [direction*.3, 120., 0., -.25, direction*.3])
    np.testing.assert_allclose(out[3, 5:10], [-direction*.3, 90., -direction*5., -.25, 0.])
    np.testing.assert_array_equal(out[3, 10:12], [2, 3])
    np.testing.assert_array_equal(out[-1, 10:12], [6, 6])


@pytest.mark.parametrize("direction", [1, -1])
def test_ring_lookback_is_inclusive_and_wraps_without_losing_current_facts(reference, direction):
    tape = [(1., 100., 0., -.125, 1., 0, 0, False),
            (-1., 90., -10., -.125, 0., 2, 1, False),
            (1., 80., 0., -.125, 1., 3, 2, False)]
    out = _ring_run(tape, capacity=2, lookback=2, direction=direction)
    np.testing.assert_array_equal(out[:, 0], 1)
    np.testing.assert_array_equal(out[:, 2], [1, 2, 2])
    np.testing.assert_array_equal(out[:, 1], [0, 0, 1])
    np.testing.assert_array_equal(out[:, 4], 0)
    np.testing.assert_array_equal(out[-1, 10:13], [2, 2, 3])


def test_ring_overflow_is_sticky_and_requires_explicit_reset(reference):
    tape = [(1., 100., 0., -.125, 1., 0, 0, False),
            (-1., 90., -10., -.125, 0., 1, 1, False),
            (1., 80., 0., -.125, 1., 2, 2, False),
            (1., 80., 0., -.125, 2., 20, 3, False),
            (1., 80., 0., -.125, 1., 21, 0, True)]
    out = _ring_run(tape, capacity=2)
    np.testing.assert_array_equal(out[:, 0], [1, 1, 0, 0, 1])
    np.testing.assert_array_equal(out[:, 4], [0, 0, 1, 1, 0])
    np.testing.assert_array_equal(out[2:4, 5:13], np.tile(out[1, 5:13], (2, 1)))
    np.testing.assert_array_equal(out[-1, 1:3], [0, 1])


def test_ring_merge_at_capacity_preserves_cancelling_cashflows(reference):
    tape = [(-1., 100., 16777216., -.125, 1., 1, 0, False),
            (-1., 90., -16777216., -.125, 0., 1, 1, False)]
    out = _ring_run(tape, capacity=1)
    np.testing.assert_array_equal(out[:, 0], 1)
    np.testing.assert_array_equal(out[:, 2], 1)
    np.testing.assert_array_equal(out[-1, 5:10], [-2., 100., 0., -.25, 0.])


@pytest.mark.parametrize("change", ["minute", "sequence", "quantity", "price", "position"])
def test_ring_rejects_malformed_or_reordered_producer_facts(reference, change):
    first = (1., 100., 0., -.125, 1., 2, 3, False)
    second = [1., 100., 0., -.125, 2., 3, 4, False]
    index, value = {"minute": (5, 1), "sequence": (6, 3), "quantity": (0, 0.),
                    "price": (1, float("nan")), "position": (4, -2.)}[change]
    second[index] = value
    out = _ring_run([first, tuple(second)])
    assert out[-1, 0] == 0
    assert out[-1, 4] == 2
    np.testing.assert_array_equal(out[-1, 5:13], out[0, 5:13])


def test_ring_does_not_merge_through_an_actual_flat(reference):
    tape = [(-1., 100., -10., -.125, 0., 1, 0, False),
            (-1., 90., -20., -.125, 0., 1, 1, False)]
    out = _ring_run(tape)
    np.testing.assert_array_equal(out[:, 2], [1, 2])


def test_disabled_ring_accepts_without_storing_or_retaining_state(reference):
    out = _ring_run([(1., 100., 0., -.125, 1., 0, 0, False)], enabled=False)
    assert out[0, 0] == 1
    assert out[0, 2] == out[0, 3] == out[0, 4] == 0


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("start", [0, 2, 3])
def test_compacted_ring_view_matches_original_retained_rust_samples(reference, direction, before, start):
    tape = [(1., 100., 0., -.125, 1., 0, 0, False),
            (2., 120., 0., -.125, 3., 0, 1, False),
            (-1., 90., -20., -.125, 2., 2, 2, False),
            (-2., 80., -60., -.125, 0., 2, 3, False),
            (1., 80., 0., -.125, 1., 3, 4, False)]
    _, states, records = _ring_run(tape, direction=direction, buffers=True)
    device = gpu_device()
    params = torch.tensor([[direction, 80., 85., 1., direction]], dtype=torch.float32, device=device)
    options = torch.tensor([[8, start, 3, before]], dtype=torch.int32, device=device)
    events = torch.empty((1, 16, 4), dtype=torch.float32, device=device)
    samples = torch.full((1, 16, 4), float("nan"), dtype=torch.float32, device=device)
    opening = torch.full((1, 2), float("nan"), dtype=torch.float32, device=device)
    _ring_library().ring_history_probe(states, records, params, options, events, samples, opening, threads=1)
    payload = dict(start=start, end=3, fills_before_same_time_price=before,
                   position=dict(size=direction, basis=80., mark=85., multiplier=1.,
                                 quantity_step=1., inverse=False, pside="long" if direction==1 else "short"),
                   fills=[dict(identity=str(sequence), timestamp=minute, delta=direction*qty,
                               price=price, realized=direction*gross, fee=fee, sequence=sequence, revision=0)
                          for qty, price, gross, fee, after, minute, sequence, reset in tape],
                   prices={str(t):95. for t in range(start, 4)})
    ref = json.loads(reference.hsl_history(json.dumps(payload)))
    expected = [[v[k] for k in ("size", "basis", "pnl", "upnl")] for v in ref["samples"]]
    np.testing.assert_allclose(samples.cpu().numpy()[0, :len(expected)], expected, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(opening.cpu().numpy()[0], [ref["opening_size"], ref["opening_basis"]], rtol=2e-6, atol=2e-5)
