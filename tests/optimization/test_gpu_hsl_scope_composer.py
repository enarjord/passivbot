"""Fresh GPU scope observations and permissions match source-verified Rust."""

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

M = 60_000
FACTS = 128
PRICES = 160
POINTS = 512


@pytest.fixture(scope="module")
def reference():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    return passivbot_rust


_PROBE = r"""
kernel void scope_composer_probe(
    device const float* params, device const int* info, device const float* fills,
    device const int* chronology, device const float* prices, device HslPairEvent* events,
    device HslScopePoint* points, device float* output, device int* status,
    uint b [[thread_position_in_grid]]
) {
    int io=int(b)*10, po=int(b)*24, count=info[io];
    HslScopePair pairs[3];
    for(int p=0;p<count;++p) {
        int base=(int(b)*3+p)*128;
        thread HslScopePair& pair=pairs[p];
        pair.facts.values=reinterpret_cast<device const HslPairFact*>(fills+base*4);
        pair.facts.minutes=chronology+base*2;
        pair.facts.head=info[io+4+p*2]; pair.facts.count=info[io+5+p*2];
        pair.facts.capacity=128; pair.facts.value_stride=1; pair.facts.minute_stride=2;
        pair.sequences=chronology+base*2+1; pair.sequence_stride=2;
        pair.prices=prices+(int(b)*3+p)*160; pair.price_stride=1;
        pair.candles=nullptr; pair.candle_stride=0; pair.fallback_price=0.0f;
        pair.current_size=params[po+p*6]; pair.current_basis=params[po+p*6+1];
        pair.current_mark=params[po+p*6+2]; pair.multiplier=params[po+p*6+3];
        pair.short_side=params[po+p*6+4]<0.0f;
        pair.events=events+base;
        if(!hsl_reconstruct_pair(pair.facts,pair.current_size,pair.current_basis,
            pair.short_side,params[po+p*6+5],pair.events,pair.history)) {
            status[int(b)]=-1; return;
        }
    }
    HslScopeResult result;
    bool valid=hsl_compose_scope(pairs,count,info[io+1],info[io+2],info[io+3]!=0,
        params[po+18],params[po+19],params[po+20],params[po+21],params[po+22]!=0.0f,
        result,params[po+23]!=0.0f ? points+int(b)*512 : nullptr,512);
    status[int(b)]=valid ? 1 : -2;
    int out=int(b)*8;
    output[out]=result.raw; output[out+1]=result.ema; output[out+2]=float(result.action);
    output[out+3]=float(result.flat_minute); output[out+4]=float(result.point_count);
    output[out+5]=float(result.latest_flat_minute);
    output[out+6]=result.latest_flat_raw; output[out+7]=result.latest_flat_ema;
}
"""


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    source = "\n".join((root / name).read_text() for name in
                       ("mps_hsl_history.metal", "mps_hsl_scope.metal"))
    return compile_shader("#include <metal_stdlib>\nusing namespace metal;\n" + source + _PROBE)


def _run(rows):
    n = len(rows)
    params = np.zeros((n, 24), np.float32)
    info = np.zeros((n, 10), np.int32)
    fills = np.zeros((n, 3, FACTS, 4), np.float32)
    chronology = np.zeros((n, 3, FACTS, 2), np.int32)
    prices = np.zeros((n, 3, PRICES), np.float32)
    for b, row in enumerate(rows):
        info[b, :4] = [len(row["pairs"]), row["start"], row["end"], row["before"]]
        params[b, 18:23] = [row["budget"], row["span"], row["threshold"],
                            row["cooldown"], row["never"]]
        params[b, 23] = row.get("trace", True)
        for p, pair in enumerate(row["pairs"]):
            pos = pair["position"]
            params[b, p*6:p*6+6] = [pos[k] for k in ("size", "basis", "mark", "multiplier")] + [
                -1 if pos["pside"] == "short" else 1, pos["quantity_step"]]
            selected = [f for f in pair["fills"] if row["start"] <= f["timestamp"] <= row["end"]]
            assert len(selected) <= FACTS
            head = (b*7+p*11) % FACTS
            info[b, 4+p*2:6+p*2] = [head, len(selected)]
            for i, fill in enumerate(selected):
                slot = (head+i) % FACTS
                fills[b, p, slot] = [fill[k] for k in ("delta", "price", "realized", "fee")]
                chronology[b, p, slot] = [fill["timestamp"], fill["sequence"]]
            prices[b, p, :row["end"]-row["start"]+1] = [
                pair["prices"][str(t)] for t in range(row["start"], row["end"]+1)]
    device = gpu_device()
    inputs = [torch.from_numpy(a).to(device) for a in (params, info, fills, chronology, prices)]
    scratch = torch.empty((n, 3, FACTS, 4), device=device)
    trace_rows = POINTS if any(row.get("trace", True) for row in rows) else 1
    points = torch.full((n, trace_rows, 8), float("nan"), device=device)
    output = torch.empty((n, 8), device=device)
    status = torch.empty(n, dtype=torch.int32, device=device)
    _library().scope_composer_probe(*inputs, scratch, points, output, status, threads=n)
    return points.cpu().numpy(), output.cpu().numpy(), status.cpu().numpy()


def _request(row):
    pairs = []
    for pair in row["pairs"]:
        pairs.append(dict(
            symbol=pair["symbol"], position=pair["position"], position_at=row["end"]*M,
            mark_at=row["end"]*M, fills_started_at=0, fills_at=row["end"]*M,
            prices_at=row["end"]*M, revisions=[0, 0, 0, 0], fills_position_anchor=None,
            fills=[dict(f, timestamp=f["timestamp"]*M) for f in pair["fills"]],
            prices={str(int(t)*M):v for t, v in pair["prices"].items()},
        ))
    return dict(now=row["end"]*M, start=row["start"]*M, balance=row["budget"],
                balance_at=row["end"]*M, config_at=row["end"]*M, max_current_age_ms=0,
                mode="unified", pside=None, symbol=None, global_fill_sequence=True,
                fills_before_same_time_price=row["before"], pairs=pairs)


def _check(reference, rows):
    points, results, statuses = _run(rows)
    np.testing.assert_array_equal(statuses, 1)
    for b, row in enumerate(rows):
        episodes = json.loads(reference.hsl_trace(json.dumps(_request(row))))["episodes"]
        expected = [p for episode in episodes for p in episode["points"]]
        decisions = json.loads(reference.hsl_controller(json.dumps(dict(
            episodes=episodes, now=row["end"]*M, start=row["start"]*M, budget=row["budget"],
            span=row["span"], threshold=row["threshold"], cooldown_ms=row["cooldown"]*M,
            restart="never" if row["never"] else "always"))))
        assert int(results[b, 4]) == len(expected), row["label"]
        final = decisions[-1]
        actions = {"normal":0, "halted":1, "panic":3}
        np.testing.assert_allclose(results[b, :2], [final["raw"], final["ema"]],
                                   atol=2e-6, rtol=2e-5, err_msg=row["label"])
        assert int(results[b, 2]) == actions[final["action"]]
        assert int(results[b, 3]) == (-1 if final["flat_at"] is None else final["flat_at"]//M)
        flats = [i for i, point in enumerate(expected) if point["flatten"]]
        if flats:
            index = flats[-1]
            assert int(results[b, 5]) == expected[index]["timestamp"]//M
            np.testing.assert_allclose(results[b, 6:8], [decisions[index]["raw"], decisions[index]["ema"]],
                                       atol=2e-6, rtol=2e-5, err_msg=row["label"])
        else:
            np.testing.assert_array_equal(results[b, 5:], [-1., 0., 0.])
        if not row.get("trace", True):
            assert points.shape[1] == 1
            assert np.isnan(points[b]).all()
            continue
        actual = points[b, :len(expected)]
        integers = actual.view(np.int32)
        np.testing.assert_array_equal(integers[:, 0], [p["timestamp"]//M for p in expected],
                                      err_msg=row["label"])
        np.testing.assert_array_equal(integers[:, 3:5],
                                      [[p["exposed"], p["flatten"]] for p in expected],
                                      err_msg=row["label"])
        np.testing.assert_allclose(actual[:, 1:3], [[p["pnl"], p["upnl"]] for p in expected],
                                   atol=3e-4, rtol=2e-5, err_msg=row["label"])
        np.testing.assert_allclose(actual[:, 5:7], [[p["raw"], p["ema"]] for p in decisions],
                                   atol=2e-6, rtol=2e-5, err_msg=row["label"])
        np.testing.assert_array_equal(integers[:, 7], [actions[p["action"]] for p in decisions],
                                      err_msg=row["label"])


def _cases(before, span, never):
    rng = random.Random(7129)
    rows = []
    for seed in range(24):
        pairs = []
        seq = 0
        for p, sign in enumerate((1, -1, 1)):
            size, basis, tape = 0, 0., []
            for minute in range(1, 33):
                for _ in range(1 + int(minute % 7 == 0)):
                    q = -size if size and minute % 8 == 0 else (
                        rng.randrange(1, 5) if not size or rng.random() < .55 else -rng.randrange(1, size+1))
                    price = float(rng.randrange(80, 121))
                    gross = 0. if q > 0 else -q*sign*(price-basis)
                    if q > 0:
                        basis = (size*basis+q*price)/(size+q)
                    size += q
                    if not size:
                        basis = 0.
                    tape.append(dict(identity=str(seq), timestamp=minute, delta=sign*q,
                                     price=price, realized=gross, fee=-.125, sequence=seq, revision=0))
                    seq += 1
            pairs.append(dict(symbol=str(p), position=dict(
                size=sign*size, basis=basis, mark=85., multiplier=1., quantity_step=1.,
                inverse=False, pside="long" if sign == 1 else "short"), fills=tape,
                prices={str(t):float(90+(t*7+p*3)%23) for t in range(35)}))
        # Sequence must be globally ordered by simulator execution, including
        # different pairs and distinct flat/reopen runs within one minute.
        ordered = sorted([f for p in pairs for f in p["fills"]],
                         key=lambda f:(f["timestamp"], f["sequence"]))
        for seq, f in enumerate(ordered):
            f["identity"] = str(seq)
            f["sequence"] = seq
        selected = pairs[:1] if seed % 3 == 0 else (pairs[::2] if seed % 3 == 1 else pairs)
        rows.append(dict(label=f"seed={seed},before={before},span={span},never={never}",
                         pairs=selected, start=(0, 3, 16, 29)[seed % 4], end=34, before=before,
                         budget=1000., span=span, threshold=.13, cooldown=2, never=never))
    return rows


@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("span", [1., 2.5, 25.])
@pytest.mark.parametrize("never", [False, True])
def test_fresh_scope_matches_rust_observations_and_permissions(reference, before, span, never):
    _check(reference, _cases(before, span, never))


def test_compact_scope_result_requires_no_point_trace(reference):
    rows = _cases(False, 2.5, True)
    for row in rows:
        row["trace"] = False
    _check(reference, rows)


@pytest.mark.parametrize("short,held", [(False, False), (False, True), (True, True)])
def test_empty_history_current_snapshot(reference, short, held):
    pair = dict(symbol="A", position=dict(size=(-1 if short else 1) if held else 0.,
                basis=100. if held else 0., mark=80. if not short else 120., multiplier=1.,
                quantity_step=1., inverse=False, pside="short" if short else "long"),
                fills=[], prices={str(t):100. for t in range(5)})
    row = dict(label="empty", pairs=[pair], start=0, end=4, before=False,
               budget=100., span=25., threshold=.1, cooldown=2, never=False)
    _check(reference, [row])


@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("kind", ["closed", "reopened", "missing_close", "current_opening", "round_trips"])
@pytest.mark.parametrize("trace", [True, False])
def test_terminal_and_reopening_accounting(reference, before, sign, kind, trace):
    adverse = 80. if sign == 1 else 120.
    tape = [(1, 2., 100., 0.), (2, -1., adverse, -20.), (3, -1., adverse, -20.)]
    size, basis = 0., 0.
    if kind == "reopened":
        tape.append((4, 1., adverse, 0.))
        size, basis = sign, adverse
    elif kind == "missing_close":
        tape = tape[:2]
    elif kind == "current_opening":
        size, basis = sign, 100.
    elif kind == "round_trips":
        tape.extend([(3, 1., 100., 0.), (3, -1., adverse, -20.),
                     (3, 1., 100., 0.), (3, -1., adverse, -20.)])
    fills = [dict(identity=str(i), timestamp=t, delta=sign*q, price=p, realized=g,
                  fee=-1., sequence=i, revision=0) for i, (t, q, p, g) in enumerate(tape)]
    pair = dict(symbol="A", position=dict(size=size, basis=basis, mark=adverse,
                multiplier=1., quantity_step=1., inverse=False, pside="long" if sign == 1 else "short"),
                fills=fills, prices={str(t):100. if t < 2 else adverse for t in range(7)})
    rows = [dict(label=f"{kind},sign={sign},before={before},span={span}", pairs=[pair],
                 start=0, end=6, before=before, budget=100., span=span,
                 threshold=.1, cooldown=100, never=True, trace=trace) for span in (1., 2.5, 25.)]
    _check(reference, rows)


@pytest.mark.parametrize("never", [False, True])
@pytest.mark.parametrize("cooldown", [0, 2, 1000])
def test_compact_flat_tail_preserves_endpoint_expiry_and_observation_count(reference, never, cooldown):
    rows = []
    for sign in (1, -1):
        adverse = 80. if sign == 1 else 120.
        fills = [dict(identity=str(i), timestamp=t, delta=sign*q, price=p, realized=g,
                      fee=-1., sequence=i, revision=0)
                 for i, (t, q, p, g) in enumerate(
                     [(1, 2., 100., 0.), (2, -1., adverse, -20.), (3, -1., adverse, -20.)])]
        pair = dict(symbol="A", position=dict(size=0., basis=0., mark=adverse,
                    multiplier=1., quantity_step=1., inverse=False,
                    pside="long" if sign == 1 else "short"), fills=fills,
                    prices={str(t):100. if t < 2 else adverse for t in range(140)})
        for start in (0, 2, 4):
            for before in (False, True):
                for span in (1., 2.5, 25.):
                    rows.append(dict(label=f"idle sign={sign},start={start},before={before},span={span}",
                                     pairs=[pair], start=start, end=139, before=before,
                                     budget=100., span=span, threshold=.1, cooldown=cooldown,
                                     never=never, trace=False))
    _check(reference, rows)
