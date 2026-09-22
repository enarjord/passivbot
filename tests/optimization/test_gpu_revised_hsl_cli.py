"""Offline public GPU optimization with exact revised Rust validation."""
import json
import os
import pickle
import sys

import pytest
from test_hsl_revised_offline_runtime import deny_network, offline_cli_config

torch=pytest.importorskip('torch')
pytestmark=pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason='GPU unavailable',
)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['coin','pside','unified'])
@pytest.mark.parametrize('coin_count',[1,2])
async def test_revised_gpu_optimizer_cli_is_offline(tmp_path,monkeypatch,mode,coin_count):
    from optimize import main
    from optimization.shape import build_optimization_shape
    from config.optimize_bounds import set_flat_optimize_bound
    import msgpack

    cfg=offline_cli_config(tmp_path,monkeypatch,mode)
    if coin_count==2:
        add_offline_coin(tmp_path,cfg)
    cfg['live']['strategy_kind']='trailing_martingale'
    cfg['optimize'].update(backend='gpu',iters=32,n_cpus=1,population_size=4,
        scoring=[{'metric':'adg_usd','goal':'max'}],limits=[],seed=42,
        enable_overrides=[],compress_results_file=False,write_all_results=True)
    cfg['optimize']['gpu'].update(population_size=4,batch_size=4,exact_workers=1,
        max_pending_exact=2,validate_per_generation=2,drift_probes=1)
    shape=build_optimization_shape(cfg)
    for key,path in shape.key_paths:
        value=cfg
        for part in path:
            value=value[part]
        set_flat_optimize_bound(cfg['optimize']['bounds'],'trailing_martingale',key,[value,value])
    target=cfg['optimize']['bounds'] if mode=='unified' else cfg['optimize']['bounds']['long']
    target['hsl']['red_threshold']=[.01,.2]
    guard=tmp_path/'guard';guard.mkdir()
    (guard/'sitecustomize.py').write_text(
        "import socket\n_connect=socket.socket.connect\n"
        "def denied(*a, **k): raise RuntimeError('offline GPU optimizer attempted network')\n"
        "def connect(self,address):\n"
        "    if self.family != socket.AF_UNIX: return denied()\n"
        "    return _connect(self,address)\n"
        "socket.socket.connect=connect\nsocket.getaddrinfo=denied\n")
    monkeypatch.setenv('PYTHONPATH',str(guard)+os.pathsep+os.environ.get('PYTHONPATH',''))
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys,'argv',['optimize',str(path),'--suite','n'])
    with pytest.raises(SystemExit) as finished:
        await main()
    assert finished.value.code==0
    artifacts=list((tmp_path/'optimize_results').rglob('all_results.bin'))
    assert len(artifacts)==1
    with artifacts[0].open('rb') as f:
        records=list(msgpack.Unpacker(f,raw=False))
    assert records
    checkpoint = artifacts[0].parent / 'checkpoint.pkl'
    state = pickle.loads(checkpoint.read_bytes())
    assert state['generation'] > 0
    assert state['exact_done'] >= 32
    assert state['halt_reason'] is None
    assert state['optimizer_evaluation_contract']['live']['hsl_engine'] == 'revised'
    assert all(record['live']['hsl_engine'] == 'revised' for record in records)
    before = artifacts[0].stat().st_size
    cfg['optimize']['iters'] = 48
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, 'argv', ['optimize', str(path), '--suite', 'n',
                                    '--resume', str(artifacts[0].parent)])
    with pytest.raises(SystemExit) as finished:
        await main()
    assert finished.value.code == 0
    assert artifacts[0].stat().st_size > before
    resumed = pickle.loads(checkpoint.read_bytes())
    assert resumed['exact_done'] > state['exact_done']
    assert resumed['optimizer_evaluation_contract'] == state['optimizer_evaluation_contract']


def add_offline_coin(root,cfg):
    """Duplicate the synthetic fixture under a second public symbol, offline."""
    from copy import deepcopy
    import numpy as np
    from ohlcv_catalog import OhlcvCatalog
    from ohlcv_store import OhlcvStore
    from test_simulation_offline import START
    from test_hsl_revised_offline_runtime import runtime_inputs
    cache=root/'caches'
    for path in cache.rglob('*.json'):
        values=json.loads(path.read_text())
        if path.name=='markets.json':
            market=deepcopy(values['BTC/USDT:USDT'])
            market.update(id='ETHUSDT',symbol='ETH/USDT:USDT',base='ETH')
            values['ETH/USDT:USDT']=market
        elif 'BTC' in values:
            values['ETH']=deepcopy(values['BTC'])
            if path.name.endswith('_symbols.json'):
                values['ETH']={'binanceusdm':'ETH/USDT:USDT'}
        path.write_text(json.dumps(values))
    cfg['live']['approved_coins']['long']=['BTC','ETH']
    _,_,candles,_=runtime_inputs(cfg['live']['hsl_signal_mode'])
    stamps=START+np.arange(-1,1441,dtype=np.int64)*60000
    rows=np.concatenate([candles[:1,0],candles[:,0],
                         np.repeat(candles[-1:,0],1441-len(candles),axis=0)])
    rows[:,:3]*=1.1
    store=OhlcvStore(cache/'ohlcvs',OhlcvCatalog(cache/'ohlcvs/catalog.sqlite'))
    store.write_rows('binance','1m','ETH/USDT:USDT',stamps,rows.astype(np.float32))
