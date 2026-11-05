"""Bounded native HSL wrapper benchmark, with complete Decision parity first.

Run from the repository root with a source-matched extension:
PYTHONPATH=src:tests python tests/benchmarks/hsl_json.py --baseline BASE_REF
The baseline is local repository source; no exchange data or requests are used.
"""
import argparse
import dataclasses
import statistics
import subprocess
import sys
import time
import types
from test_hsl_runtime import bot, quotes, capture, NOW, SYMBOL
from test_hsl_inputs import event
from live import hsl_runtime as current
from live import hsl_continuity as continuity


def module_at(name, path):
    source = subprocess.check_output(['git', 'show', args.baseline+':'+path], text=True)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, path, 'exec'), module.__dict__)
    return module
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", required=True)
parser.add_argument("--cohort", type=int, default=50)
parser.add_argument("--fills", type=int, default=500)
parser.add_argument("--samples", type=int, default=7)
parser.add_argument("--iterations", type=int, default=10)
args = parser.parse_args()
if min(args.cohort, args.samples, args.iterations) < 1 or args.fills < 0:
    parser.error("counts must be positive and fills nonnegative")
old_continuity = module_at('baseline_continuity', 'src/live/hsl_continuity.py')
old = module_at('baseline_runtime', 'src/live/hsl_runtime.py')
old.FactualReplays = old_continuity.FactualReplays
old.continuity_status = old_continuity.continuity_status
fills = [event(id=str(i), timestamp=NOW - (args.fills-i)*60_000, side='buy' if i%2==0 else 'sell',
               qty=0.01 if i%2==0 else -0.01, price=100., pnl=0., c_mult=1.) for i in range(args.fills)]
base, unavailable = capture(bot(events=fills), quotes(), {}, symbols={'long':[SYMBOL], 'short':[SYMBOL]},
                           now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000)
assert not unavailable
request = base[0]
cohort = args.cohort
old_requests = tuple(dataclasses.replace(request, replay=old_continuity.ScopeReplay(request.metadata, request.price_grids)) for _ in range(cohort))
new_requests = tuple(dataclasses.replace(request, replay=continuity.ScopeReplay(request.metadata, request.price_grids)) for _ in range(cohort))
assert [dataclasses.asdict(x) for x in old.evaluate(old_requests)] == [dataclasses.asdict(x) for x in current.evaluate(new_requests)]
rows = {'baseline':[], 'current':[]}
for rep in range(args.samples):
    for name, runtime, requests in [('baseline',old,old_requests), ('current',current,new_requests)]:
        t=time.process_time()
        for _ in range(args.iterations): runtime.evaluate(requests)
        rows[name].append(time.process_time()-t)
print('requests_per_batch',cohort,'fill_events',len(fills),'metadata_bytes',len(request.metadata))
for name, values in rows.items():print(name,'median_cpu_s',statistics.median(values),'samples',values)
print('ratio',statistics.median(rows['current'])/statistics.median(rows['baseline']))
