"""Offline execution of actual shared HSL callers; no device speed claim."""
from pathlib import Path
import shutil
import subprocess

import pytest

_HEADER = r'''
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#define thread
#define device
#define constant const
#define kernel
#define MAX_COINS 2
#define PASSIVBOT_HSL_CAPACITY 34
#define PASSIVBOT_HSL_TREE_SIZE 32
#define PASSIVBOT_HSL_LOOKBACK 32
#define PASSIVBOT_HSL_NATIVE_CACHE_CAPACITY 128
#define PASSIVBOT_HSL_FACTUAL_ONLY 1
#define PASSIVBOT_HSL_FACTS_ENABLED 1
struct float2 {float x,y;float2(float a,float b):x(a),y(b){} float2()=default;};
using uint=unsigned int;using ulong=unsigned long;
using std::isfinite;using std::min;using std::max;
// Keep float shader calls in F32 on both standard-library implementations.
using std::fmax;using std::fmin;
static_assert(sizeof(fmax(0.0f,1.0f))==sizeof(float), "fmax must preserve F32");
static_assert(sizeof(fmin(0.0f,1.0f))==sizeof(float), "fmin must preserve F32");
template<class T>T clamp(T v,T lo,T hi){return std::max(lo,std::min(v,hi));}
int reconstruction_visits=0,suffix_hits=0,cache_hits=0,forced_declines=0;
bool forced_checked=false;
'''
_CALLER = r'''
int main() {
    std::cout << std::hexfloat;
    const int length=240;
    float params[]={1,.02f,2.5f,5,0,0,2};
    float settings[]={.125f,0,0,0,1,0,0,length-1,.125f,0,0,0,1,0,0,length-1};
    float bars[length*8]={};
    for(int t=0;t<length;++t) for(int c=0;c<2;++c) for(int k=0;k<3;++k)
        bars[t*8+c*4+k]=100.0f+float((t+c*7)%19)*.125f;
    HslState aggregate=load_hsl(params,0,0),coins[2];
    for(int c=0;c<2;++c) coins[c]=load_hsl(params,0,0);
    int nodes=hsl_storage_nodes(34,32,64);
    std::vector<HslNode> trees(3*nodes);
    int words=0;
#if PASSIVBOT_HSL_NATIVE_CACHE_ENABLED
    words=hsl_native_cache_bytes(2,64)/4;
#endif
    std::vector<int> rows(max(words,1));
    bind_hsl_multicoin_hsl(aggregate,coins,trees.data(),rows.data(),0,2,true,true,64);
    HslReplayContext context=hsl_replay_context(bars,settings,2,8,true);
    attach_hsl_replay_side(context,aggregate,coins,0,false);
    int sequence=0;
    float budget=10000.0f;
    auto fill=[&](int c,int t,float delta) {
        float old=coins[c].facts.state->current_size,basis=coins[c].facts.state->current_basis;
        float price=bars[t*8+c*4+2];
        float size=old+delta;
        float realized=delta<0 ? -delta*(price-basis):0;
        if(delta>0) basis=old==0?price:fma(delta/size,price-basis,basis);
        if(size==0) basis=0;
        HslPairFact fact={delta,price,realized,-.001f};
        if(!hsl_pair_ring_capture(coins[c].facts,fact,t,sequence++,size,basis,false)) return false;
        budget+=realized-.001f;return true;
    };
    auto query=[&](const char* tag,int t,bool terminal=false) {
        bool exposed=coins[0].facts.state->current_size!=0 || coins[1].facts.state->current_size!=0;
        observe_hsl(aggregate,budget,0,0,exposed,t,terminal);
        std::cout<<tag<<" "<<t<<" "<<aggregate.hsl_valid<<" "<<aggregate.hsl.raw<<" "
            <<aggregate.hsl.ema<<" "<<aggregate.hsl.action<<" "<<aggregate.hsl.flat_minute<<" "
            <<aggregate.hsl.last_observed<<" "<<aggregate.triggers<<" "<<aggregate.restarts
            <<" "<<suffix_hits<<" "<<reconstruction_visits<<" "<<cache_hits<<"\n";
    };
    for(int t=0;t<length;++t) {
        if(t>0) query("walk",t);
#if PASSIVBOT_HSL_NATIVE_CACHE_ENABLED
        if(forced_declines==1 && !forced_checked) {
            forced_checked=true;
            HslNativeCache cache=hsl_native_cache_view(rows.data(),2,64);
            if(!cache.header->valid)return 10;
            // A numerical query decline invokes independent logical scratch.
            // Reinitialization must restore physical ownership before reuse.
            for(int c=0;c<2;++c) {
                int first=max(t-32,cache.header->cutoff.found?cache.header->cutoff.minute:0);
                HslPairFacts view=hsl_pair_ring_view_after(coins[c].facts,first,cache.header->cutoff);
                HslPairEvent expected[64];HslPairHistory history;
                if(!hsl_reconstruct_pair(view,coins[c].facts.state->current_size,
                    coins[c].facts.state->current_basis,false,.125f,expected,history))return 11;
                for(int i=0;i<view.count;++i) {
                    HslPairEvent actual=coins[c].fact_events[hsl_pair_slot(view,i)];
                    if(actual.before!=expected[i].before || actual.after!=expected[i].after
                        || actual.basis!=expected[i].basis)return 12;
                }
            }
        }
#endif
        if(t==0) {if(!fill(0,t,8))return 2;}
        if(t<200 && t!=160) {
            float size=coins[1].facts.state->current_size;
            // Dense round trips clip and wrap. An all-increase interval has no
            // certified flat and must decline until a proven reset is retained.
            float delta=(t>=60 && t<100)? .5f : size==0?8.0f:t%3==1?-2.0f:-size;
            if(!fill(1,t,delta))return 3;
        }
        if(t==0) query("cold",t);
        if(t==45) query("same_minute",t);
        if(t==110) {
            float size=coins[1].facts.state->current_size;
            if(size!=0 && !fill(1,t,-size))return 4;
            if(!fill(1,t,8))return 5;
            query("flat_reopen",t);
        }
        if(t==135) {
#if PASSIVBOT_HSL_NATIVE_CACHE_ENABLED
            aggregate.native_cache_reset=true;
#endif
            query("cache_loss",t);
        }
        if(t==160) {
            for(int c=0;c<2;++c) {
                float size=coins[c].facts.state->current_size;
                if(size!=0 && !fill(c,t,-size))return 6;
            }
            query("terminal",t,true);
            if(!fill(0,t,8) || !fill(1,t,8))return 7;
            query("terminal_reopen",t,true);
        }
        if(t==200) {
            bool failed=false;
            for(int i=0;i<70;++i) {
                float size=coins[1].facts.state->current_size;
                float delta=i%2==0?1.0f:-1.0f;
                if(!fill(1,t,delta)) {failed=true;break;}
            }
            if(!failed || coins[1].facts.state->failure!=1)return 8;
            query("overflow",t);
            break;
        }
    }
    if(coins[1].facts.state->head==0)return 9;
    std::cout<<"counts "<<reconstruction_visits<<" "<<suffix_hits<<" "<<cache_hits<<" "<<forced_declines<<"\n";
}
'''


def _source():
    root = Path(__file__).resolve().parents[2] / 'passivbot-rust/src/gpu'
    source = '\n'.join((root / name).read_text() for name in (
        'mps_hsl.metal', 'mps_hsl_history.metal', 'mps_hsl_scope.metal',
        'mps_hsl_native_cache.metal', 'mps_hsl_common.metal'))
    marker = ') {\n    float direction = short_side ? -1.0f : 1.0f;\n    if (facts.capacity'
    assert source.count(marker) == 1
    source = source.replace(marker, ') {\n    reconstruction_visits += 2*facts.count;\n    float direction = short_side ? -1.0f : 1.0f;\n    if (facts.capacity')
    marker = 'const int width = HSL_NATIVE_CACHE_BLOCK;'
    assert source.count(marker) == 1
    source = source.replace(marker, 'if(end>35 && forced_declines==0) {++forced_declines;return false;}\n    '+marker)
    marker = 'pair.history = prefix.history;'
    assert source.count(marker) == 1
    source = source.replace(marker, '++suffix_hits;\n    '+marker)
    marker = 'cache.header->valid = 1;\n    return accepted;'
    assert source.count(marker) == 1
    return source.replace(marker, 'cache.header->valid = 1;\n    cache_hits += accepted;\n    return accepted;')


def _compile_run(tmp_path, name, cache, suffix, retire=True):
    compiler = shutil.which('clang++') or shutil.which('c++')
    if compiler is None:
        pytest.skip('Offline shader caller proof requires a C++ compiler')
    path, executable = tmp_path / f'{name}.cpp', tmp_path / name
    definitions = (f'#define PASSIVBOT_HSL_NATIVE_CACHE_ENABLED {cache}\n'
                   f'#define PASSIVBOT_HSL_NATIVE_CACHE_FLAT_SUFFIX_ENABLED {suffix}\n')
    source = _source()
    if not retire:
        marker = 'if (cache_eligible) native_cache.header->valid = 0;'
        assert source.count(marker) == 1
        source = source.replace(marker, '// Negative control: retain overwritten physical ownership.')
    path.write_text(_HEADER + definitions + source + _CALLER)
    subprocess.run([compiler, '-std=c++17', '-O0', '-ffp-contract=off', str(path),
                    '-o', str(executable)], check=True, capture_output=True, text=True, timeout=60)
    rows = subprocess.run([str(executable)], check=True, capture_output=True,
                          text=True, timeout=20).stdout.splitlines()
    return [r.split() for r in rows[:-1]], tuple(map(int, rows[-1].split()[1:]))


def test_actual_caller_flat_suffix_wrap_and_declines(tmp_path):
    variants = [_compile_run(tmp_path, f'caller-{cache}-{suffix}', cache, suffix)
                for cache, suffix in ((0,0),(1,0),(1,1))]
    reference = variants[0][0]
    for rows, counts in variants[1:]:
        assert [r[:3]+r[5:10] for r in rows] == [r[:3]+r[5:10] for r in reference]
        for actual, expected in zip(rows, reference):
            for a, b in zip(actual[3:5], expected[3:5]):
                assert abs(float.fromhex(a)-float.fromhex(b)) <= 2e-6
        assert counts[2] > 0
    assert all(r[2] == ('0' if r[0]=='overflow' else '1') for r in reference)
    assert variants[2][1][3] == 1
    # The same real caller catches stale physical ownership if the numerical
    # decline invalidation is removed; the probe exits at its event check.
    with pytest.raises(subprocess.CalledProcessError) as error:
        _compile_run(tmp_path, 'caller-stale-owner', 1, 1, retire=False)
    assert error.value.returncode == 12
    assert variants[2][1][1] > 20
    assert variants[2][1][0] < variants[1][1][0]
    optimized = {int(r[1]): r for r in variants[2][0] if r[0] == 'walk'}
    # After the final old flat expires, positive-only facts provide no reset.
    # The actual caller takes fresh reconstruction instead of inventing one.
    assert int(optimized[100][10]) == int(optimized[96][10])
    assert int(optimized[100][11]) > int(optimized[96][11])
    assert int(optimized[100][12]) > int(optimized[96][12])
    assert int(optimized[59][10]) > int(optimized[40][10])



def test_actual_empty_scope_expiry_and_new_fill_with_cache(tmp_path):
    from test_gpu_hsl_empty_native import _CALLER as empty_caller

    compiler = shutil.which('clang++') or shutil.which('c++')
    if compiler is None:
        pytest.skip('Offline shader caller proof requires a C++ compiler')
    root = Path(__file__).resolve().parents[2] / 'passivbot-rust/src/gpu'
    source = '\n'.join((root / name).read_text() for name in (
        'mps_hsl.metal', 'mps_hsl_history.metal', 'mps_hsl_scope.metal',
        'mps_hsl_native_cache.metal', 'mps_hsl_common.metal'))
    header = (_HEADER.replace('PASSIVBOT_HSL_CAPACITY 34', 'PASSIVBOT_HSL_CAPACITY 16')
                     .replace('PASSIVBOT_HSL_TREE_SIZE 32', 'PASSIVBOT_HSL_TREE_SIZE 1')
                     .replace('PASSIVBOT_HSL_LOOKBACK 32', 'PASSIVBOT_HSL_LOOKBACK 10'))
    caller = empty_caller.replace(' [[thread_position_in_grid]]', '')
    main = r"""
int main() {
    float params[]={1,.002f,2.5f,5,0,0,1};
    float settings[]={1,0,0,0,1,0,0,24};
    float bars[25*4]={};
    for(int t=0;t<25;++t)for(int k=0;k<3;++k)
        bars[t*4+k]=t<11?100:t<15?80:t<18?90:70;
    std::vector<HslNode> trees(2*26);
    int words=1;
#if PASSIVBOT_HSL_NATIVE_CACHE_ENABLED
    words=hsl_native_cache_bytes(1,16)/4;
#endif
    std::vector<int> rows(words);
    float out[25*7];
    std::fill(out,out+25*7,NAN);
    int closing=0;
    empty_native_caller(params,bars,settings,trees.data(),rows.data(),out,closing,0);
    std::cout<<std::hexfloat;
    for(int t=0;t<25;++t){for(int k=0;k<7;++k)std::cout<<out[t*7+k]<<" ";std::cout<<"\n";}
}
"""
    results = []
    for enabled in (0,1):
        path, exe = tmp_path/f'empty-{enabled}.cpp', tmp_path/f'empty-{enabled}'
        path.write_text(header + f'#define PASSIVBOT_HSL_NATIVE_CACHE_ENABLED {enabled}\n'
                        + source + caller + main)
        subprocess.run([compiler,'-std=c++17','-O0','-ffp-contract=off',str(path),'-o',str(exe)],
                       check=True,capture_output=True,text=True,timeout=60)
        output = subprocess.run([str(exe)],check=True,capture_output=True,text=True,timeout=20)
        results.append([[float.fromhex(v) for v in row.split()]
                        for row in output.stdout.splitlines()])
    assert results[0] == results[1]
    assert all(row[0] == 1 for row in results[1])
    assert all(row[5] == 0 for row in results[1][11:15])
    assert results[1][15][5] > 0
    assert results[1][14][3] == results[1][18][3] == 3
