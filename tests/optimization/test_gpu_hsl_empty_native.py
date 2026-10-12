"""The empty-history shortcut preserves native caller and simulation behavior."""
import numpy as np
import pytest


def _device_reference():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('NVIDIA CUDA required')
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    info = verify_loaded_runtime_extension()
    assert not info.get('skipped')
    return torch, passivbot_rust


_CALLER = r"""
kernel void empty_native_caller(constant float* params, constant float* bars,
    constant float* settings, device HslNode* trees, device int* rows,
    device float* out, constant int& closing, uint b [[thread_position_in_grid]]) {
    HslState aggregate=load_hsl(params,0,0), coins[1];
    coins[0]=load_hsl(params,0,0);
    bind_hsl_multicoin_hsl(aggregate,coins,trees,rows,0,1,true,true,16);
    HslReplayContext context=hsl_replay_context(bars,settings,1,8,true);
    attach_hsl_replay_side(context,aggregate,coins,0,false);
    float budget=1000.0f;
    int sequence=0;
    for(int t=0;t<25;++t) {
        if(t==0 || t==15) {
            float old=coins[0].facts.state->current_size;
            float old_basis=coins[0].facts.state->current_basis;
            float price=bars[t*4+2];
            float basis=old==0.0f ? price : fma(1.0f/(old+1.0f),price-old_basis,old_basis);
            HslPairFact fact={1.0f,price,0.0f,-0.01f};
            if(!hsl_pair_ring_capture(coins[0].facts,fact,t,sequence++,old+1.0f,basis,false)) return;
            budget-=0.01f;
        }
        bool terminal=t==18 && closing!=0;
        if(terminal) {
            float size=coins[0].facts.state->current_size;
            float price=bars[t*4+2];
            float realized=size*(price-coins[0].facts.state->current_basis);
            HslPairFact fact={-size,price,realized,-0.01f};
            if(!hsl_pair_ring_capture(coins[0].facts,fact,t,sequence++,0.0f,0.0f,false)) return;
            budget+=realized-0.01f;
        }
        bool exposed=coins[0].facts.state->current_size!=0.0f;
        observe_hsl(aggregate,budget,0.0f,0.0f,exposed,t,terminal);
        int o=t*7;
        out[o]=aggregate.hsl_valid;
        out[o+1]=aggregate.hsl.raw;out[o+2]=aggregate.hsl.ema;
        out[o+3]=aggregate.hsl.action;out[o+4]=aggregate.hsl.flat_minute;
        out[o+5]=hsl_pair_ring_view(coins[0].facts,max(t-10,0)).count;
        out[o+6]=aggregate.hsl.last_observed;
    }
}
"""


@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
@pytest.mark.parametrize('closing',[False,True],ids=['mark-control','terminal-fill'])
def test_native_caller_empty_expiry_new_fill_and_terminal_ablation(strategy,closing):
    torch, reference=_device_reference()
    from optimization.gpu.runtime import compile_shader, gpu_device
    from optimization.gpu.mps_kernel import _hsl_native_cache_bytes
    device=gpu_device()
    candles=np.zeros((25,4),dtype=np.float32)
    candles[:,:3]=100.0
    candles[11:15,:3]=80.0
    candles[15:18,:3]=90.0
    candles[18:,:3]=70.0
    bars=torch.as_tensor(candles,device=device)
    settings=torch.tensor([1,0,0,0,1,0,0,24],dtype=torch.float32,device=device)
    params=torch.tensor([1,.002,2.5,5,0,0,1],dtype=torch.float32,device=device)
    source=getattr(reference,f'mps_{strategy}_multicoin_source_py')()
    source=('#define PASSIVBOT_HSL_FACTUAL_ONLY 1\n'
            '#define PASSIVBOT_HSL_FACTS_ENABLED 1\n'
            '#define PASSIVBOT_HSL_CAPACITY 16\n'
            '#define PASSIVBOT_HSL_TREE_SIZE 1\n'
            '#define PASSIVBOT_HSL_LOOKBACK 10\n'+source+_CALLER)
    outputs=[]
    variants=[(0,0),(1,0)]
    if strategy=="ema_anchor" and not closing:
        variants.append((1,1))
    for enabled,cache in variants:
        trees=torch.zeros((2,26,32),dtype=torch.uint8,device=device)
        words=_hsl_native_cache_bytes(128,1,16)//4 if cache else 0
        rows=torch.empty(words,dtype=torch.int32,device=device)
        out=torch.full((25,7),float('nan'),dtype=torch.float32,device=device)
        library=compile_shader(f'#define PASSIVBOT_HSL_EMPTY_SCOPE_ENABLED {enabled}\n'
            f'#define PASSIVBOT_HSL_NATIVE_CACHE_ENABLED {cache}\n'
            '#define PASSIVBOT_HSL_NATIVE_CACHE_CAPACITY 128\n'+source)
        library.empty_native_caller(params,bars,settings,trees,rows,out,int(closing),threads=1)
        outputs.append(out.cpu().numpy())
    for output in outputs[1:]:
        np.testing.assert_array_equal(outputs[0],output)
        assert np.all(output[:,0]==1)
    # Opening fact expires while actual inventory is still held; a later fill
    # restores retained history. The mark-only and genuine close paths differ.
    assert np.all(outputs[1][11:15,5]==0)
    assert outputs[1][15,5]>0
    assert outputs[1][14,3]==3
    assert outputs[1][18,3]==(1 if closing else 3)
    assert outputs[1][18,4]==(18 if closing else -1)


@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
def test_native_full_replay_empty_compile_ablation_preserves_all_outputs(monkeypatch,strategy):
    torch,_=_device_reference()
    import backtest
    from optimization.gpu import mps_kernel
    from test_gpu_hsl_multicoin import make_proxy,raw

    def forbidden(*args,**kwargs):
        raise AssertionError('CPU backtest used by native replay')
    monkeypatch.setattr(backtest,'execute_backtest',forbidden)
    compiler=mps_kernel.compile_shader
    libraries=(mps_kernel._ema_anchor_multicoin_shader_library,
               mps_kernel._trailing_martingale_multicoin_shader_library)
    results=[]
    try:
        for enabled in (0,1):
            for library in libraries:
                library.cache_clear()
            monkeypatch.setattr(mps_kernel,'compile_shader',
                lambda source,*args,_enabled=enabled,**kwargs: compiler(
                    f'#define PASSIVBOT_HSL_EMPTY_SCOPE_ENABLED {_enabled}\n'+source,*args,**kwargs))
            # Full replay smoke coverage uses the supported public lookback.
            # This short fixture does not reach expiry; actual caller tests
            # above cover expiry with a bounded shader-level physical window.
            proxy=make_proxy('unified',strategy,('long',),minutes=384,
                             lookback=1,factual_hsl=True)
            _,output=raw(proxy,[{}])
            results.append(output)
        assert results[0].keys()==results[1].keys()
        assert results[1]['fill_count'].item()>0
        for key,value in results[0].items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,results[1][key],rtol=0,atol=0,equal_nan=True,msg=key)
            else:
                assert value==results[1][key],key
    finally:
        for library in libraries:
            library.cache_clear()
