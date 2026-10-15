"""Empty factual shortcuts preserve actual coin/pside caller observations."""
from functools import lru_cache
import numpy as np
import pytest

_CALLER=r"""
kernel void empty_selected_caller(constant float* params, constant float* bars,
    constant float* settings, device HslNode* trees, device int* rows,
    device float* out, constant int& closing, uint b [[thread_position_in_grid]]) {
    int po=int(b)*7;
    HslState aggregate=load_hsl(params,po,0), coins[2];
    coins[0]=load_hsl(params,po,0); coins[1]=load_hsl(params,po,0);
    bool short_side=(int(b)&1)!=0;
    float direction=short_side ? -1.0f : 1.0f;
    bind_hsl_multicoin_hsl(aggregate,coins,trees,rows,int(b)*3,2,true,true,16);
    constant float* case_bars=bars+int(b)*25*8;
    HslReplayContext context=hsl_replay_context(case_bars,settings,2,8,true);
    attach_hsl_replay_side(context,aggregate,coins,0,short_side);
    thread HslState& selected=aggregate.signal_mode==HSL_SIGNAL_COIN ? coins[0] : aggregate;
    selected.budget_multiplier=aggregate.signal_mode==HSL_SIGNAL_COIN ? 1.5f : 1.0f;
    float balance=1000.0f;
    int sequence=0;
    for(int t=0;t<25;++t) {
        for(int c=0;c<2;++c) {
            int later=c==0 || aggregate.signal_mode!=HSL_SIGNAL_COIN ? 15 : 12;
            if(t==0 || t==later) {
                float old=coins[c].facts.state->current_size;
                float old_basis=coins[c].facts.state->current_basis;
                float price=case_bars[t*8+c*4+2];
                float basis=old==0.0f ? price : fma(1.0f/(fabs(old)+1.0f),price-old_basis,old_basis);
                HslPairFact fact={direction,price,0.0f,-0.01f};
                if(!hsl_pair_ring_capture(coins[c].facts,fact,t,sequence++,old+direction,basis,short_side)) return;
                balance-=0.01f;
            }
        }
        // Other fills or a changed slot headroom may alter current budget.
        // Empty reconstruction must use it immediately, not a prior signal.
        if(t==13) balance*=0.4f;
        bool terminal=t==18 && closing!=0;
        if(terminal) for(int c=0;c<2;++c) {
            float size=coins[c].facts.state->current_size;
            float price=case_bars[t*8+c*4+2];
            float realized=direction*fabs(size)*(price-coins[c].facts.state->current_basis);
            HslPairFact fact={-size,price,realized,-0.01f};
            if(!hsl_pair_ring_capture(coins[c].facts,fact,t,sequence++,0.0f,0.0f,short_side)) return;
            balance+=realized-0.01f;
        }
        bool exposed=coins[0].facts.state->current_size!=0.0f;
        if(aggregate.signal_mode!=HSL_SIGNAL_COIN)
            exposed=exposed || coins[1].facts.state->current_size!=0.0f;
        // Each side has its own adverse mark grid. Both directions are tested;
        // historical/current prices remain valid facts rather than mocked PNL.
        observe_hsl(selected,balance,0.0f,0.0f,exposed,t,terminal);
        int o=(int(b)*25+t)*10;
        out[o]=selected.hsl_valid;
        out[o+1]=selected.hsl.raw; out[o+2]=selected.hsl.ema;
        out[o+3]=selected.hsl.action;out[o+4]=selected.hsl.flat_minute;
        out[o+5]=hsl_pair_ring_view(coins[0].facts,max(t-10,0)).count;
        out[o+6]=hsl_pair_ring_view(coins[1].facts,max(t-10,0)).count;
        out[o+7]=selected.hsl.last_observed;
        out[o+8]=selected.triggers;out[o+9]=selected.restarts;
    }
}
"""


@lru_cache(maxsize=4)
def _library(strategy,enabled):
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader
    source=getattr(passivbot_rust,f'mps_{strategy}_multicoin_source_py')()
    source=(f'#define PASSIVBOT_HSL_EMPTY_SCOPE_ENABLED {enabled}\n'
            '#define PASSIVBOT_HSL_FACTUAL_ONLY 1\n'
            '#define PASSIVBOT_HSL_FACTS_ENABLED 1\n'
            '#define PASSIVBOT_HSL_CAPACITY 16\n'
            '#define PASSIVBOT_HSL_TREE_SIZE 1\n'
            '#define PASSIVBOT_HSL_LOOKBACK 10\n'+source+_CALLER)
    return compile_shader(source,cuda_coin_capacity=2,mps_coin_capacity=2)


@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
def test_selected_empty_caller_expiry_unrelated_fill_budget_and_terminal(strategy):
    from test_gpu_hsl_empty_native import _device_reference
    torch,_=_device_reference()
    from optimization.gpu.runtime import gpu_device
    device=gpu_device()
    # All policy/direction/terminal inputs share just two compile variants per
    # strategy. Direction toggles within the grid; modes remain runtime inputs.
    params=torch.tensor([[1,.1,2.5,5,0,mode,2] for mode in (1,2) for _ in (0,1)],
                        dtype=torch.float32,device=device)
    candles=np.zeros((4,25,2,4),dtype=np.float32)
    candles[:,:,:,:3]=100.
    for b in range(4):
        direction=-1 if b&1 else 1
        candles[b,11:15,:,:3]=100.-direction*20.
        candles[b,15:18,:,:3]=100.-direction*10.
        candles[b,18:,:,:3]=100.-direction*30.
    bars=torch.as_tensor(candles,device=device)
    settings=torch.tensor([[1,0,0,0,1,0,0,24]]*2,dtype=torch.float32,device=device)
    for closing in (0,1):
        results=[]
        for enabled in (0,1):
            trees=torch.zeros((4,3,26,32),dtype=torch.uint8,device=device)
            rows=torch.empty(0,dtype=torch.int32,device=device)
            out=torch.full((4,25,10),float('nan'),dtype=torch.float32,device=device)
            _library(strategy,enabled).empty_selected_caller(params,bars,settings,trees,rows,out,closing,threads=4)
            results.append(out.cpu().numpy())
        np.testing.assert_array_equal(results[0],results[1])
        assert np.all(results[1][:,:,0]==1)
        # Inclusive retained edge; then empty actual coin/pside histories.
        assert np.all(results[1][:,10,5:7]>0)
        assert np.all(results[1][:,11,5:7]==0)
        # Coin0 is independent of coin1's later activity. Its empty signal must
        # change immediately when the current raw-balance budget changes.
        assert np.all(results[1][2:,12:15,5]==0)
        assert np.all(results[1][2:,12:15,6]>0)
        assert np.all(results[1][2:,13,1]!=results[1][2:,12,1])
        assert np.all(results[1][:,15,5]>0)
        assert np.all(results[1][:,18,4]==(18 if closing else -1))
        if closing: assert np.all(results[1][:,18,3]==1)
