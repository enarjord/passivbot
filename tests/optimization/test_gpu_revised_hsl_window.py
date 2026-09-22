"""Hardware parity for revised rolling arithmetic; optimizer dispatch stays gated."""
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optimization.gpu.runtime import compile_shader, gpu_device

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Apple MPS and NVIDIA CUDA unavailable",
)


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).resolve().parents[2].joinpath(
        "passivbot-rust/src/gpu/mps_hsl_revised.metal"
    ).read_text()
    return compile_shader("#include <metal_stdlib>\nusing namespace metal;\n" + source + r"""
kernel void revised_window_probe(
    constant float* inputs [[buffer(0)]],
    constant int* minutes [[buffer(1)]],
    constant int* settings [[buffer(2)]],
    device RevisedHslNode* trees [[buffer(3)]],
    device int* times [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint b [[thread_position_in_grid]]
) {
    const int steps = settings[0], capacity = settings[1], tree_size = settings[2];
    device RevisedHslNode* tree = trees + int(b) * tree_size * 2;
    device int* clock = times + int(b) * capacity;
    RevisedHslWindow w = revised_hsl_init(tree, capacity, tree_size, inputs[int(b)*steps*4+3]);
    for (int i = 0; i < steps; ++i) {
        int o = (int(b) * steps + i) * 4;
        int t = minutes[i];
        revised_hsl_expire(w, tree, clock, t - settings[3]);
        bool valid = revised_hsl_push(w, tree, clock, t, inputs[o] + inputs[o+1]);
        float2 result = valid ? revised_hsl_signal(w, tree, inputs[o+2] - inputs[o], -INFINITY)
            : float2(NAN, NAN);
        int dst = (int(b)*steps+i)*2;
        output[dst] = result.x;
        output[dst+1] = result.y;
    }
}
""")


def run_window(inputs, minutes, lookback):
    candidates, steps, _ = inputs.shape
    capacity = min(lookback + 1, steps)
    tree_size = 1 << (capacity - 1).bit_length()
    device = gpu_device()
    args = [
        torch.tensor(inputs, dtype=torch.float32, device=device),
        torch.tensor(minutes, dtype=torch.int32, device=device),
        torch.tensor([steps, capacity, tree_size, lookback], dtype=torch.int32, device=device),
        torch.empty((candidates, 2 * tree_size, 32), dtype=torch.uint8, device=device),
        torch.empty((candidates, capacity), dtype=torch.int32, device=device),
        torch.empty((candidates, steps, 2), dtype=torch.float32, device=device),
    ]
    library().revised_window_probe(*args, threads=(candidates, 1, 1))
    return args[-1].cpu().numpy()


@pytest.mark.parametrize("lookback", [1, 17, 173, 2000])
@pytest.mark.parametrize("span", [1., 2.5, 308., 10000.5])
def test_changed_anchor_expiration_and_same_minute_match_rust(lookback, span):
    import passivbot_rust
    steps = 1200
    rng = np.random.default_rng(42)
    minutes = np.cumsum(rng.choice([0, 1, 1, 1, 3], steps)).astype(np.int32)
    realized = np.cumsum(rng.integers(-5, 6, (4, steps)), axis=1)
    upnl = rng.integers(-300, 301, (4, steps))
    budget = rng.choice([100., 500., 2000.], (4, steps))
    inputs = np.stack([realized, upnl, budget, np.full((4, steps), span)], axis=-1).astype(np.float32)
    actual = run_window(inputs, minutes, lookback)
    assert np.isfinite(actual).all()
    for b in range(4):
        for i in sorted(set([0, 1, steps - 1, *range(0, steps, 31)])):
            rows = [(int(minutes[j])*60000, float(realized[b,j]), float(upnl[b,j]))
                    for j in range(i+1) if minutes[j] >= minutes[i] - lookback]
            ref = json.loads(passivbot_rust.hsl_revised_signal(
                rows, float(budget[b,i]), span, .1, None))
            np.testing.assert_allclose(actual[b,i], [ref['raw'][-1], ref['ema'][-1]],
                                       rtol=2e-5, atol=2e-6, err_msg=f"candidate={b} step={i}")


@pytest.mark.parametrize("direction", [-1., 1.])
def test_monotone_extremes_and_full_window_expiration(direction):
    n = 4096
    values = (np.arange(n) * direction).astype(np.float32)
    data = np.zeros((1, n, 4), np.float32)
    data[0,:,1] = values
    data[0,:,2] = 10000
    data[0,:,3] = 356.5
    out = run_window(data, np.arange(n, dtype=np.int32), 1440)
    assert np.isfinite(out).all()
    if direction > 0:
        np.testing.assert_array_equal(out, 0)
    else:
        import passivbot_rust
        rows = [(i*60000, 0., float(values[i])) for i in range(n-1441, n)]
        ref = json.loads(passivbot_rust.hsl_revised_signal(rows, 10000., 356.5, .1, None))
        np.testing.assert_allclose(out[0,-1], [ref['raw'][-1], ref['ema'][-1]], rtol=2e-5, atol=2e-6)


@lru_cache(maxsize=1)
def controller_library():
    source = Path(__file__).resolve().parents[2].joinpath(
        "passivbot-rust/src/gpu/mps_hsl_revised.metal").read_text()
    return compile_shader("#include <metal_stdlib>\nusing namespace metal;\n" + source + r"""
kernel void revised_controller_probe(
    constant float* inputs [[buffer(0)]], constant int* flags [[buffer(1)]],
    constant float* policy [[buffer(2)]], device RevisedHslNode* tree [[buffer(3)]],
    device int* times [[buffer(4)]], device float* realized [[buffer(5)]],
    device float* output [[buffer(6)]], uint b [[thread_position_in_grid]]
) {
    RevisedHslController h = revised_hsl_controller_init(tree, 64, 64, policy[0]);
    for (int i = 0; i < int(policy[4]); ++i) {
        bool valid = revised_hsl_observe(h, tree, times, realized,
            flags[i*3], int(policy[5]), inputs[i*3], inputs[i*3+1], inputs[i*3+2],
            flags[i*3+1] != 0, flags[i*3+2] != 0, policy[1], policy[2], policy[3] > 0.0f);
        output[i*3] = valid ? float(h.action) : -1.0f;
        output[i*3+1] = h.raw;
        output[i*3+2] = h.ema;
    }
}
""")


def run_controller(events, span, never, lookback=100):
    device = gpu_device()
    # event = minute, balance budget, realized, upnl, exposed, terminal
    values = torch.tensor([[e[1], e[2], e[3]] for e in events], device=device, dtype=torch.float32)
    flags = torch.tensor([[e[0], e[4], e[5]] for e in events], device=device, dtype=torch.int32)
    policy = torch.tensor([span, .1, 10, never, len(events), lookback], device=device)
    tree = torch.empty((128, 32), device=device, dtype=torch.uint8)
    times = torch.empty(64, device=device, dtype=torch.int32)
    realized = torch.empty(64, device=device)
    output = torch.empty((len(events), 3), device=device)
    controller_library().revised_controller_probe(values, flags, policy, tree, times,
                                                 realized, output, threads=1)
    return output.cpu().numpy()


@pytest.mark.parametrize("span", [1., 3.5])
@pytest.mark.parametrize("never", [False, True])
def test_current_red_terminal_cooldown_reclassification_and_reopen(span, never):
    import passivbot_rust
    events = [
        (0, 1000, 0, 0, False, False),
        (1, 1000, -1, -20, True, False),
        (2, 1000, -1, -500, True, False),
        (3, 750, -251, -250, True, False),  # partial panic close
        (4, 750, -251, 100, True, False),  # recovery revokes panic
        (5, 850, -151, 0, False, True),     # profitable terminal, no cooldown
        (6, 850, -152, -10, True, False),
        (7, 850, -152, -600, True, False),
        (8, 250, -752, 0, False, True),
        (9, 250, -752, 0, False, False),
        (10, 10000, -752, 0, False, False), # deposit reclassifies terminal signal
        (11, 250, -752, 0, False, False),   # restored cooldown uses original flat
        (12, 250, -753, 0, True, False),    # exposure clears previous cooldown
        (13, 250, -753, -150, True, False),
        (14, 100, -903, 0, False, True),
        (24, 100, -903, 0, False, False),
    ]
    actual = run_controller(events, span, never)
    episodes, points, opened = [], [], None
    for i, e in enumerate(events):
        t, budget, pnl, upnl, exposed, terminal = e
        if not points and episodes:
            seed = dict(episodes[-1]['points'][-1], flatten=False)
            points = [seed]
        if episodes and exposed and opened is None:
            opened = t*60000
        points.append(dict(timestamp=t*60000, pnl=pnl, upnl=upnl,
                           exposed=exposed, flatten=terminal))
        episode = dict(points=points, opened_at=opened)
        observed = episodes + [episode]
        payload = dict(episodes=observed, now=t*60000, start=-6000000,
                       budget=budget, span=span, threshold=.1,
                       cooldown_ms=600000, restart='never' if never else 'always')
        expected = json.loads(passivbot_rust.hsl_revised_controller(json.dumps(payload)))[-1]
        code = {'normal': 0, 'halted': 1, 'panic': 3}[expected['action']]
        np.testing.assert_allclose(actual[i], [code, expected['raw'], expected['ema']],
                                   atol=2e-6, rtol=2e-5, err_msg=f"event={i}")
        if terminal:
            episodes.append(episode)
            points, opened = [], None


def test_never_cooldown_is_forgotten_when_terminal_leaves_lookback():
    events = [(0,1000,0,0,False,False), (1,1000,0,-400,True,False),
              (2,600,-400,0,False,True), (3,600,-400,0,False,False),
              (12,600,-400,0,False,False), (13,600,-400,0,False,False)]
    actual = run_controller(events, 1., True, lookback=10)
    assert actual[2,0] == 1
    assert actual[-1,0] == 0
