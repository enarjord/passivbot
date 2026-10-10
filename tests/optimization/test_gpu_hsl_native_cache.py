"""Experimental bounded unified replay cache against the independent native path."""
import numpy as np
import pytest


def _device_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    assert not verify_loaded_runtime_extension().get("skipped")
    return torch, passivbot_rust


_CALLER = r"""
kernel void native_cache_caller(constant float* params, constant float* bars,
    constant float* settings, device HslNode* trees, device int* rows,
    device float* out, constant int& reset_at, uint b [[thread_position_in_grid]]) {
    const int count=2, length=3000;
    HslState aggregate=load_hsl(params,int(b)*7,0), coins[2];
    for(int c=0;c<count;++c) coins[c]=load_hsl(params,int(b)*7,0);
    bind_hsl_multicoin_hsl(aggregate,coins,trees,rows,int(b)*3,count,true,true,256);
    HslReplayContext context=hsl_replay_context(bars,settings,count,8,true);
    attach_hsl_replay_side(context,aggregate,coins,0,false);
    int sequence=0;
    float budget=10000.0f;
    for(int t=0;t<length;++t) {
        if(t==reset_at) {
#if PASSIVBOT_HSL_NATIVE_CACHE_ENABLED
            // Losing arithmetic has no trading authority. Exercise the actual
            // owner reset after a long-lived cache and physical ring wrap.
            aggregate.native_cache_reset=true;
#endif
        }
        if(t>0) observe_hsl(aggregate,budget,0.0f,0.0f,true,t,false);
        if((t==0 || t%13==0 || t%17==0 || t%800==799)
            && bars[t*8+2]>0.0f && bars[t*8+6]>0.0f) {
            int first=t==0 ? 0 : 1;
            for(int c=first;c<count;++c) {
                float old=coins[c].facts.state->current_size;
                float basis=coins[c].facts.state->current_basis;
                float price=bars[t*8+c*4+2];
                float delta=t==0 ? 5.0f : t%800==799 ? -old : t%13==0 ? 1.0f : -0.5f;
                if(delta<0.0f) delta=-fmin(-delta,old);
                float size=old+delta;
                float realized=delta<0.0f ? -delta*(price-basis) : 0.0f;
                if(delta>0.0f) basis=old==0.0f ? price : fma(delta/size,price-basis,basis);
                if(size==0.0f) basis=0.0f;
                HslPairFact fact={delta,price,realized,-0.01f};
                if(!hsl_pair_ring_capture(coins[c].facts,fact,t,sequence++,size,basis,false)) return;
                budget+=realized-0.01f;
                if(size==0.0f) {
                    HslPairFact reopen={5.0f,price,0.0f,-0.02f};
                    if(!hsl_pair_ring_capture(coins[c].facts,reopen,t,sequence++,5.0f,price,false)) return;
                    budget-=0.02f;
                }
            }
        }
        if(t==0) observe_hsl(aggregate,budget,0.0f,0.0f,true,t,false);
        if(t==1450) observe_hsl(aggregate,budget,0.0f,0.0f,true,t,false);
        if(t==2400) {
            for(int c=0;c<count;++c) {
                float size=coins[c].facts.state->current_size;
                float price=bars[t*8+c*4+2];
                float realized=size*(price-coins[c].facts.state->current_basis);
                HslPairFact close={-size,price,realized,-0.01f};
                if(!hsl_pair_ring_capture(coins[c].facts,close,t,sequence++,0.0f,0.0f,false)) return;
                budget+=realized-0.01f;
            }
            observe_hsl(aggregate,budget,0.0f,0.0f,false,t,true);
        }
        int o=(int(b)*length+t)*6;
        out[o]=aggregate.hsl_valid;out[o+1]=aggregate.hsl.raw;out[o+2]=aggregate.hsl.ema;
        out[o+3]=aggregate.hsl.action;out[o+4]=aggregate.hsl.flat_minute;
        out[o+5]=aggregate.hsl.last_observed;
        if(t==2400) {
            for(int c=0;c<count;++c) {
                float price=bars[t*8+c*4+2];
                HslPairFact reopen={5.0f,price,0.0f,-0.02f};
                if(!hsl_pair_ring_capture(coins[c].facts,reopen,t,sequence++,5.0f,price,false)) return;
                budget-=0.02f;
            }
        }
    }
}
"""


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_caller_busy_rolling_wrap_and_cache_loss(strategy):
    _native_caller_ablation(strategy)


def test_native_caller_dense_flat_suffix():
    _native_caller_ablation("ema_anchor", dense=True)


def _native_caller_ablation(strategy, dense=False):
    torch, reference = _device_reference()
    from optimization.gpu.runtime import compile_shader, gpu_device
    from optimization.gpu.mps_kernel import _hsl_native_cache_bytes
    device = gpu_device()
    length = 3000
    candles = np.zeros((length, 2, 4), dtype=np.float32)
    for c in range(2):
        candles[:, c, :3] = (100 + 10*np.sin((np.arange(length)+c*13)*.021))[:, None]
    candles[1550,:,:3] = 0.0
    bars = torch.as_tensor(candles, device=device)
    settings = torch.tensor([[1,0,0,0,1,0,0,length-1]]*2, dtype=torch.float32, device=device)
    params = torch.tensor([[1,.002,2.5,5,0,0,2], [1,.01,10000,5,0,0,2]],
                          dtype=torch.float32, device=device)
    caller = _CALLER
    if dense:
        schedule = "(t==0 || t%13==0 || t%17==0 || t%800==799)"
        assert caller.count(schedule) == 1
        caller = caller.replace(schedule, "true")
        caller = caller.replace("t==0 ? 5.0f : t%800==799 ? -old : t%13==0 ? 1.0f : -0.5f",
                                "t==0 ? 5.0f : old==0.0f ? 5.0f : t%3==1 ? -2.0f : -old")
    lookback = 64 if dense else 1440
    source = ("#define PASSIVBOT_HSL_FACTUAL_ONLY 1\n"
              "#define PASSIVBOT_HSL_FACTS_ENABLED 1\n"
              "#define PASSIVBOT_HSL_CAPACITY 1442\n"
              "#define PASSIVBOT_HSL_TREE_SIZE 32\n"
              f"#define PASSIVBOT_HSL_LOOKBACK {lookback}\n"
              "#define PASSIVBOT_HSL_NATIVE_CACHE_CAPACITY 1536\n"
              + getattr(reference, f"mps_{strategy}_multicoin_source_py")() + caller)
    outputs = []
    for enabled in (0, 1):
        library = compile_shader(f"#define PASSIVBOT_HSL_NATIVE_CACHE_ENABLED {enabled}\n"+source)
        for reset in (-1, 1900):
            trees = torch.empty((2,3,386,32), dtype=torch.uint8, device=device)
            words = _hsl_native_cache_bytes(1536,2,256)//4 if enabled else 0
            rows = torch.empty((2,words), dtype=torch.int32, device=device)
            out = torch.full((2,length,6), float("nan"), dtype=torch.float32, device=device)
            library.native_cache_caller(params,bars,settings,trees,rows,out,reset,threads=2)
            outputs.append(out.cpu().numpy())
    np.testing.assert_array_equal(outputs[0], outputs[1])
    for output in outputs[2:]:
        assert np.all(output[:,:,0] == 1)
        np.testing.assert_array_equal(outputs[0][:,:,[0,3,4,5]], output[:,:,[0,3,4,5]])
        np.testing.assert_allclose(outputs[0][:,:,1:3], output[:,:,1:3], rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("long","short")])
def test_native_full_replay_cache_ablation_and_temporal_boundary(monkeypatch, strategy, sides):
    torch, _ = _device_reference()
    import backtest
    from test_gpu_hsl_multicoin import make_proxy, raw, compare

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU backtest used by native replay")
    monkeypatch.setattr(backtest,"execute_backtest",forbidden)
    outputs = []
    for cache in (False, True):
        proxy = make_proxy("unified",strategy,sides,minutes=1800,factual_hsl=True)
        runner = proxy.fused_runner or proxy.runners[sides[0]]
        runner.hsl_native_cache_enabled = cache
        _, output = raw(proxy,[{}])
        outputs.append(output)
        if cache:
            assert runner._hsl_native_cache_capacity() > 0
            assert runner._hsl_native_cache_bytes_per_candidate() > 0
    assert outputs[0]["fill_count"].item() > 0
    compare(outputs[0],outputs[1])
    # Persisted replay state carries the arithmetic cursor across dispatches;
    # every new attempt resets it. Fused replay does not support this split yet.
    if len(sides) == 1:
        proxy = make_proxy("unified",strategy,sides,minutes=1800,factual_hsl=True,chunk=True)
        runner = proxy.runners[sides[0]]
        runner.hsl_native_cache_enabled = True
        _, output = raw(proxy,[{}])
        compare(outputs[0],output)
        _, repeated = raw(proxy,[{}])
        compare(output,repeated)


def test_owner_scratch_admission_and_compile_ablation():
    pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import MpsEmaAnchorMulticoinRunner, _hsl_native_cache_bytes
    runner = object.__new__(MpsEmaAnchorMulticoinRunner)
    from types import SimpleNamespace
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner
    from optimization.gpu.model import EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    from optimization.gpu.entry_intervals import entry_interval_history_bytes
    runner.bars = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    runner.dispatch_native_hsl_cache_eligible = True
    runner.hsl_native_cache_enabled = False
    runner.hsl_capacity = 90*1440+2
    runner.native_factual_hsl = True
    runner.hsl_fact_capacity = 4096
    runner.hsl_replay_sides = 2
    runner.n_coins = 25
    runner.hsl_scopes = 52
    before = runner._hsl_history_bytes_per_candidate()
    assert runner._hsl_native_cache_capacity() == 0
    runner.hsl_native_cache_enabled = True
    capacity = runner._hsl_native_cache_capacity()
    assert capacity % 64 == 0 and capacity > runner.hsl_capacity
    cache_bytes = runner._hsl_native_cache_bytes_per_candidate()
    assert cache_bytes == _hsl_native_cache_bytes(capacity,50,4096)
    assert 7*1024**2 < cache_bytes < 8*1024**2
    assert runner._hsl_history_bytes_per_candidate() == before+cache_bytes
    runner.hsl_native_cache_budget_bytes = cache_bytes-1
    assert runner._hsl_native_cache_capacity() == 0
    assert runner._hsl_history_bytes_per_candidate() == before
    runner.hsl_native_cache_budget_bytes = cache_bytes
    runner.hsl_fact_capacity *= 2
    assert runner._hsl_native_cache_capacity() == 0
    runner.hsl_native_cache_budget_bytes = 64*1024**2
    assert runner._hsl_native_cache_bytes_per_candidate() > cache_bytes
    runner.dispatch_hsl_disabled = True
    assert runner._hsl_native_cache_bytes_per_candidate() == 0
    runner.dispatch_hsl_disabled = False
    runner.native_factual_hsl = False
    assert runner._hsl_native_cache_bytes_per_candidate() == 0

    runner.native_factual_hsl = True
    runner.dispatch_hsl_disabled = False
    runner.hsl_scratch_budget_bytes = 512*1024**2
    runner.unstuck_pnl_capacity = 0
    runner.max_dispatch_candidate_bars = None
    runner._replay_state_bytes = None
    runner.n, runner.n_days = 3000, 3
    runner.raw_strategy_risk_enabled = runner.raw_strategy_growth_enabled = False
    runner.weighted_equity_metrics = frozenset()
    runner.recovery_distribution_enabled = runner.hsl_ema_tail_enabled = False
    runner.entry_interval_enabled = True
    base = runner._history_bytes_per_candidate() - runner._hsl_native_cache_bytes_per_candidate()
    assert base == runner.hsl_scopes * (2+8192+(8192+1)//2)*32 + entry_interval_history_bytes(runner.n)
    assert runner._history_bytes_per_candidate() == base+runner._hsl_native_cache_bytes_per_candidate()
    runner.side = "long"
    runner.cuda_coin_capacity = 32
    runner.mps_coin_capacity = None
    runner.pnl_lookback_bars = 90*1440
    runner.unstuck_pnl_lookback_bars = 0
    runner.dynamic_wel_by_tradability = True
    runner.loss_gate_specialization = True
    runner.loss_gate_enabled = False
    runner.hsl_raw_drawdown_enabled = runner.hsl_raw_tail_enabled = False
    runner.btc_risk_enabled = runner.equity_balance_diff_enabled = False
    runner.weighted_volume_enabled = False
    runner.weighted_raw_equity_enabled = runner.weighted_account_equity_enabled = False
    runner.hsl_raw_tail_capacity = 1
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate()-1
    assert runner._disable_native_cache_over_budget()
    assert runner._hsl_native_cache_capacity() == 0
    assert runner._history_bytes_per_candidate() == base
    assert not runner._disable_native_cache_over_budget()
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorMulticoinFusedRunner, MpsTrailingMartingaleMulticoinFusedRunner,
    )
    for cls in (MpsEmaAnchorMulticoinRunner,MpsEmaAnchorMulticoinFusedRunner,
                MpsTrailingMartingaleMulticoinRunner,MpsTrailingMartingaleMulticoinFusedRunner):
        other = object.__new__(cls)
        other.__dict__.update(runner.__dict__)
        assert other._library_cache_call()[1][-1] == 0
    runner.hsl_scratch_budget_bytes = 512*1024**2
    runner.dispatch_native_hsl_cache_budget_disabled = False
    for device in ("mps", "cpu"):
        runner.bars.device.type = device
        assert runner._hsl_native_cache_capacity() == 0
    runner.bars.device.type = "cuda"
    assert runner._hsl_native_cache_capacity() > 0
    assert MpsEmaAnchorMulticoinRunner.hsl_native_cache_enabled is True
    assert MpsTrailingMartingaleMulticoinRunner.hsl_native_cache_enabled is False

    # Effective request modes reset the budget latch; dormant/coin-only HSL
    # does not reserve portfolio arithmetic, even with enabled coin overrides.
    runner.hsl_fact_capacity_learned = runner.hsl_fact_capacity
    runner._hsl_scratch_buffers = {}
    runner._coin_hsl_enabled_overrides = (np.array([np.nan]),)*2
    keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    matrix = np.zeros((2,len(keys)*2), dtype=np.float32)
    for side in range(2):
        matrix[:,keys.index("hsl_enabled")+side*len(keys)] = 1
    for mode in (0,1,2):
        for side in range(2):
            matrix[:,keys.index("hsl_signal_mode")+side*len(keys)] = mode
        runner.dispatch_native_hsl_cache_budget_disabled = True
        runner._prepare_native_factual_hsl(matrix)
        assert not runner.dispatch_native_hsl_cache_budget_disabled
        assert (runner._hsl_native_cache_capacity()>0) is (mode==0)
    matrix[:,keys.index("hsl_signal_mode")] = [0,2]
    runner._prepare_native_factual_hsl(matrix)
    assert runner._hsl_native_cache_capacity()>0
    for side in range(2):
        matrix[:,keys.index("hsl_enabled")+side*len(keys)] = 0
    for side in range(2):
        matrix[:,keys.index("hsl_signal_mode")+side*len(keys)] = 2
    runner._coin_hsl_enabled_overrides = (np.array([1.]),)*2
    runner._prepare_native_factual_hsl(matrix)
    assert runner._hsl_native_cache_capacity()==0 and runner.hsl_fact_capacity==8192
    runner._coin_hsl_enabled_overrides = (np.array([np.nan]),)*2
    runner._prepare_native_factual_hsl(matrix)
    assert runner._hsl_native_cache_capacity()==0 and runner.hsl_fact_capacity==0
    assert runner.hsl_fact_capacity_learned==8192
    runner.native_factual_hsl = False
    runner._prepare_native_factual_hsl(matrix)
    assert not runner.dispatch_native_hsl_cache_eligible


    # A rejected overflow attempt can outgrow optional cache admission while
    # authoritative facts still fit. Its retry has fresh scalar ABI state and
    # never returns the rejected result; the next request reconsiders admission.
    from optimization.gpu.mps_kernel import HslFactHistoryOverflow
    runner.native_factual_hsl = True
    runner.hsl_fact_capacity = runner.hsl_fact_capacity_learned = 256
    runner._replay_states = {"old": object()}
    runner._replay_state_bytes = 777
    for side in range(2):
        matrix[:,keys.index("hsl_enabled")+side*len(keys)] = 1
        matrix[:,keys.index("hsl_signal_mode")+side*len(keys)] = 0
    runner._prepare_native_factual_hsl(matrix)
    initial_total = runner._history_bytes_per_candidate()
    runner.hsl_fact_capacity = 512
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate()-1
    assert initial_total <= runner.hsl_scratch_budget_bytes
    runner.hsl_fact_capacity = 256
    attempts = []
    runner.interrupt_check = lambda: None

    def overflow_then_accept(params, **kwargs):
        attempts.append(runner.hsl_fact_capacity)
        if len(attempts) == 1:
            assert runner._hsl_native_cache_capacity() > 0
            raise HslFactHistoryOverflow("rejected factual attempt")
        assert runner.dispatch_native_hsl_cache_budget_disabled
        assert runner._hsl_native_cache_capacity() == 0
        assert runner._replay_state_bytes is None and not runner._replay_states
        assert not runner._hsl_scratch_buffers
        return {"accepted": True}

    runner._run_factual_attempt = overflow_then_accept
    assert runner.run(matrix) == {"accepted": True}
    assert attempts == [256,512]
    assert runner.last_hsl_fact_retries == 1 and runner.hsl_fact_capacity_learned == 512
    runner.hsl_scratch_budget_bytes = 512*1024**2
    runner._run_factual_attempt = lambda params, **kwargs: {"cache": runner._hsl_native_cache_capacity()}
    assert runner.run(matrix)["cache"] > 0
    assert not runner.dispatch_native_hsl_cache_budget_disabled
    assert runner.last_hsl_fact_retries == 0
