import os

os.environ["NOJIT"] = "true"
import time
import json
import pprint
import subprocess
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from procedures import dump_live_config, make_get_filepath, ts_to_date


def main():
    oj = lambda *x: os.path.join(*x)
    algorithm = "particle_swarm_optimization"
    passivbot_mode = "neat_grid"
    d0 = f"results_{algorithm}_{passivbot_mode}"
    d0 = sys.argv[1]
    if "clock" in d0:
        passivbot_mode = "clock"
    elif "neat" in d0:
        passivbot_mode = "neat_grid"
    elif "recursive" in d0:
        passivbot_mode = "recursive_grid"
    elif "static" in d0:
        passivbot_mode = "static_grid"
    else:
        raise Exception("unknown passivbot_mode")
    if "harmony_search" in d0:
        algorithm = "harmony_search"
    elif "particle_swarm_optimization" in d0:
        algorithm = "particle_swarm_optimization"
    else:
        raise Exception("unknown algorithm")
    date_now = ts_to_date(time.time())[:19]
    dump_dir = make_get_filepath(
        f"configs/extracted/{algorithm}_{passivbot_mode}_{date_now.replace(':', '_')}/"
    )
    symbols_done = set()
    for d1 in sorted(os.listdir(d0))[::-1]:
        fp = oj(d0, d1, "all_results.txt")
        symbol = d1[20:]
        if not os.path.exists(fp) or symbol in symbols_done:
            print("skipping", fp)
            continue
        symbols_done.add(symbol)
        subprocess.run(["python3", "inspect_opt_results.py", fp, "-d"])
        shutil.move(
            oj(d0, d1, "all_results_best_config.json"),
            oj(dump_dir, f"{symbol}.json"),
        )
        try:
            shutil.copy(
                oj(d0, d1, "table_best_config.txt"),
                oj(dump_dir, f"{symbol}_inspect_results.txt"),
            )
        except Exception as e:
            print("error copying", e)


if __name__ == "__main__":
    main()
