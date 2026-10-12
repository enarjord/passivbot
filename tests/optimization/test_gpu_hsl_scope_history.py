"""Factual scope prefixes: same-time order, held/clipped history and latest flats."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optimization.gpu.runtime import compile_shader, gpu_device

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="GPU required"
)

_PROBE = r"""
kernel void scope_history_probe(
    device const float* facts, device const int* metadata,
    device const float* positions, device int* states,
    device HslPairRecord* records, device int* output,
    uint b [[thread_position_in_grid]]
) {
    HslPairRing rings[3]; float sizes[3], steps[3], prior_sizes[3]; int cursors[3];
    for (int p=0;p<3;++p) {
        rings[p].state = reinterpret_cast<device HslPairRingState*>(states+p*16);
        rings[p].records = records+p*16;
        rings[p].capacity=16; rings[p].lookback=100; rings[p].enabled=true;
        hsl_pair_ring_reset(rings[p]);
        sizes[p]=positions[p*2]; steps[p]=positions[p*2+1];
    }
    for(int i=0;i<metadata[0];++i) {
        int p=metadata[1+i*3], minute=metadata[2+i*3], sequence=metadata[3+i*3];
        HslPairFact f; f.delta=facts[i*5]; f.price=facts[i*5+1];
        f.realized=facts[i*5+2]; f.fee=facts[i*5+3];
        hsl_pair_ring_append(rings[p],f,minute,sequence,facts[i*5+4],p==2);
    }
    int selected_count=metadata[100], start=metadata[101];
    HslPairRing selected[3]; float selected_sizes[3], selected_steps[3];
    for(int p=0;p<selected_count;++p) {
        int index=metadata[102+p]; selected[p]=rings[index];
        selected_sizes[p]=sizes[index]; selected_steps[p]=steps[index];
    }
    HslScopeCutoff cutoff;
    output[0]=hsl_scope_history_cutoff(selected,selected_sizes,selected_steps,
        selected_count,cursors,prior_sizes,cutoff) ? 1 : -1;
    output[1]=cutoff.found; output[2]=cutoff.minute; output[3]=cutoff.sequence;
    for(int p=0;p<selected_count;++p) {
        HslPairFacts view=hsl_pair_ring_view_after(selected[p],start,cutoff);
        output[4+p*2]=view.count;
        output[5+p*2]=view.count ? selected[p].records[view.head].first_sequence : -1;
    }
}
"""


_CACHE_PROBE = r"""kernel void scope_cutoff_cache_probe(
    device const float* facts, device const int* metadata,
    device const float* positions, device int* states,
    device HslPairRecord* records, device int* output,
    uint b [[thread_position_in_grid]]
) {
    HslPairRing rings[3]; float sizes[3], steps[3], prior[3]; int cursors[3];
    for (int p=0;p<3;++p) {
        rings[p].state = reinterpret_cast<device HslPairRingState*>(states+p*16);
        rings[p].records = records+p*16;
        rings[p].capacity=16; rings[p].lookback=10; rings[p].enabled=true;
        hsl_pair_ring_reset(rings[p]);
        sizes[p]=positions[p*2]; steps[p]=positions[p*2+1];
    }
    HslScopeCutoffCache cache; hsl_scope_cutoff_cache_reset(cache);
    for(int query=0;query<metadata[0]+3;++query) {
        if(query<metadata[0]) {
            int p=metadata[1+query*3], minute=metadata[2+query*3], seq=metadata[3+query*3];
            HslPairFact f; f.delta=facts[query*5]; f.price=facts[query*5+1];
            f.realized=facts[query*5+2]; f.fee=facts[query*5+3];
            hsl_pair_ring_append(rings[p],f,minute,seq,facts[query*5+4],p==2);
            sizes[p]=facts[query*5+4];
        }
        int selected_count=metadata[100], start=metadata[101];
        if(query>=metadata[0]) selected_count=metadata[106];
        if(query==metadata[0]+1) {
            int selected=metadata[107]; sizes[selected]=sizes[selected]==0.0f ? 0.1f : 0.0f;
        }
        if(query==metadata[0]+2) hsl_scope_cutoff_cache_reset(cache);
        HslPairRing chosen[3]; float chosen_sizes[3], chosen_steps[3]; int ids[3];
        for(int p=0;p<selected_count;++p) {
            int index=metadata[(query>=metadata[0] ? 107 : 102)+p];
            ids[p]=index; chosen[p]=rings[index]; chosen_sizes[p]=sizes[index]; chosen_steps[p]=steps[index];
        }
        HslScopeCutoff actual, expected; bool reused;
        bool ok=hsl_scope_history_cutoff_cached(chosen,chosen_sizes,chosen_steps,ids,selected_count,
            cursors,prior,cache,actual,reused);
        output[query*12]=ok; output[query*12+1]=reused;
        bool fresh=hsl_scope_history_cutoff(chosen,chosen_sizes,chosen_steps,selected_count,cursors,prior,expected);
        output[query*12+2]=fresh && ok && actual.found==expected.found
            && actual.minute==expected.minute && actual.sequence==expected.sequence;
        bool again=hsl_scope_history_cutoff_cached(chosen,chosen_sizes,chosen_steps,ids,selected_count,
            cursors,prior,cache,actual,reused);
        output[query*12+3]=again && reused;
        bool same_views=true;
        for(int p=0;p<selected_count;++p) {
            HslPairFacts a=hsl_pair_ring_view_after(chosen[p],start+query,actual);
            HslPairFacts e=hsl_pair_ring_view_after(chosen[p],start+query,expected);
            same_views=same_views && a.head==e.head && a.count==e.count;
        }
        output[query*12+4]=same_views;
        output[query*12+5]=actual.found;output[query*12+6]=actual.minute;output[query*12+7]=actual.sequence;
        // A cache hit never bypasses malformed factual-header rejection.
        int failure=chosen[0].state->failure; chosen[0].state->failure=2;
        output[query*12+8]=!hsl_scope_history_cutoff_cached(chosen,chosen_sizes,chosen_steps,ids,selected_count,
            cursors,prior,cache,actual,reused);
        chosen[0].state->failure=failure;
        int version=chosen[0].state->version; chosen[0].state->version=-1;
        output[query*12+9]=!hsl_scope_history_cutoff_cached(chosen,chosen_sizes,chosen_steps,ids,selected_count,
            cursors,prior,cache,actual,reused);
        chosen[0].state->version=version;
    }
}
"""

@lru_cache(maxsize=1)
def _library():
    source = (Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu/mps_hsl_history.metal").read_text()
    return compile_shader("#include <metal_stdlib>\nusing namespace metal;\n"+source+_PROBE+_CACHE_PROBE)


def _run(tape, sizes, selected=(0,1,2), *, step=.1, start=0):
    facts=np.zeros((32,5),np.float32); meta=np.zeros(106,np.int32)
    meta[0]=len(tape)
    for i,(pair, minute, delta, after) in enumerate(tape):
        facts[i]=[delta,100.,0.,-.125,after]
        meta[1+i*3:4+i*3]=[pair,minute,i]
    meta[100:102]=[len(selected),start]
    meta[102:102+len(selected)]=selected
    positions=np.asarray([[size,step] for size in sizes],np.float32)
    device=gpu_device()
    args=[torch.from_numpy(a).to(device) for a in (facts,meta,positions)]
    states=torch.empty((3,16),dtype=torch.int32,device=device)
    records=torch.empty((3,16,8),dtype=torch.float32,device=device)
    output=torch.empty(10,dtype=torch.int32,device=device)
    _library().scope_history_probe(*args,states,records,output,threads=1)
    return output.cpu().numpy()


# Authored global executions. The close at seq3 proves a portfolio flat; the
# reopen at seq4 shares that minute but belongs to the next episode. A later
# long close at seq6 does not flatten the portfolio while short seq5 is held.
TAPE=[(0,1,.3,.3),(1,2,.2,.2),(0,3,-.3,0.),(1,3,-.2,0.),
      (0,3,.1,.1),(2,4,-.2,-.2),(0,5,-.1,0.),(2,6,.2,0.)]


@pytest.mark.parametrize("selected, sizes, count, minute, sequence, views", [
    ((0,1,2),[0.,0.,-.2],7,3,3,[(2,4),(0,-1),(1,5)]),
    ((0,1,2),[0.,0.,0.],8,3,3,[(2,4),(0,-1),(2,5)]),
    ((0,1),[0.,0.,-.2],7,3,3,[(2,4),(0,-1)]),
    ((0,),[0.,0.,-.2],7,3,2,[(2,4)]),
    ((1,),[0.,0.,0.],8,1,-1,[(2,1)]),
    ((2,),[0.,0.,-.2],7,3,-1,[(1,5)]),
])
def test_scope_uses_actual_selected_flat_prefix(selected,sizes,count,minute,sequence,views):
    result=_run(TAPE[:count],sizes,selected)
    np.testing.assert_array_equal(result[:4],[1,1,minute,sequence])
    np.testing.assert_array_equal(result[4:4+len(views)*2],np.asarray(views).reshape(-1))


def test_exposed_clipped_prefix_retains_lookback_without_inventing_flat_seed():
    # Opening was clipped: the factual reverse walk is still exposed before
    # the first retained partial close. It cannot prove a prehistory flat.
    result=_run([(0,20,-.1,.2)], [.2,0.,0.],(0,),start=10)
    np.testing.assert_array_equal(result[:6],[1,0,-1,-1,1,0])


def test_no_selected_fills_keeps_requested_valuation_window():
    result=_run(TAPE[:2],[.3,.2,0.],(2,),start=0)
    np.testing.assert_array_equal(result[:6],[1,0,-1,-1,0,-1])


def test_same_time_multiple_completed_episodes_keep_only_latest():
    result=_run([(0,1,.1,.1),(0,1,-.1,0.),(0,1,.2,.2),(0,1,-.2,0.)],
                [0.,0.,0.],(0,))
    np.testing.assert_array_equal(result[:6],[1,1,1,1,2,2])


def test_quantum_reverse_walk_preserves_fractional_flatness():
    result=_run([(0,1,.1,.1),(0,1,.2,.3),(0,2,-.3,0.),(0,2,.1,.1)],
                [.1,0.,0.],(0,))
    np.testing.assert_array_equal(result[:6],[1,1,2,2,1,3])


@pytest.mark.parametrize("size,step", [(float('nan'),.1),(0.,0.),(0.,float('inf'))])
def test_invalid_scope_current_facts_are_unavailable(size,step):
    result=_run([], [size,0.,0.],(0,),step=step)
    assert result[0]==-1


@pytest.mark.parametrize("selected,alternate", [((0,1,2),(0,1)), ((0,),(1,)), ((2,),(2,)), ((0,1),(1,0))])
@pytest.mark.parametrize("tape", [TAPE, [(0,1,.1,.1),(0,1,.2,.3),(0,2,-.3,0.),(0,40,.1,.1),(0,41,-.1,0.)]])
def test_factual_cutoff_memo_matches_fresh_walk_and_keeps_window_clipping(tape,selected,alternate):
    facts=np.zeros((32,5),np.float32);meta=np.zeros(112,np.int32)
    meta[0]=len(tape)
    for i,(pair,minute,delta,after) in enumerate(tape):
        facts[i]=[delta,100.,0.,-.125,after];meta[1+i*3:4+i*3]=[pair,minute,i]
    meta[100:102]=[len(selected),0];meta[102:102+len(selected)]=selected
    meta[106]=len(alternate);meta[107:107+len(alternate)]=alternate
    positions=np.asarray([[0.,.1]]*3,np.float32);device=gpu_device()
    args=[torch.from_numpy(a).to(device) for a in (facts,meta,positions)]
    states=torch.empty((3,16),dtype=torch.int32,device=device)
    records=torch.empty((3,16,8),dtype=torch.float32,device=device)
    output=torch.zeros((len(tape)+3,12),dtype=torch.int32,device=device)
    _library().scope_cutoff_cache_probe(*args,states,records,output,threads=1)
    result=output.cpu().numpy()
    # Fresh canonical cutoff and clipped fact views are the reference, including
    # coalescence, pruning, unrelated fills, pair selection and unexplained exposure.
    np.testing.assert_array_equal(result[:,[0,2,3,4,8,9]],1)
    assert result[0,1]==0
    assert result[-2,1]==0  # Exposure change invalidates the factual memo.
    assert result[-1,1]==0  # Explicit discard rebuilds exactly the same prefix.
    for i,fill in enumerate(tape):
        if i>0 and fill[0] in selected:assert result[i,1]==0
        elif i>0:assert result[i,1]==1
    assert result[len(tape),1]==int(set(selected)==set(alternate))
