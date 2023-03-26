import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"

import json
import re
import pprint
import numpy as np
from prettytable import PrettyTable
import argparse
import hjson
from procedures import load_live_config, dump_live_config, make_get_filepath
from pure_funcs import (
    config_pretty_str,
    candidate_to_live_config,
    calc_scores,
    determine_passivbot_mode,
    make_compatible,
)
from njit_funcs import round_dynamic


def main():

    parser = argparse.ArgumentParser(prog="view conf", description="inspect conf")
    parser.add_argument("results_fpath", type=str, help="path to results file")
    weights_keys = [
        ("psl", "maximum_pa_distance_std_long"),
        ("pss", "maximum_pa_distance_std_short"),
        ("pml", "maximum_pa_distance_mean_long"),
        ("pms", "maximum_pa_distance_mean_short"),
        ("pll", "maximum_loss_profit_ratio_long"),
        ("pls", "maximum_loss_profit_ratio_short"),
        ("hsl", "maximum_hrs_stuck_max_long"),
        ("hss", "maximum_hrs_stuck_max_short"),
        ("erl", "minimum_eqbal_ratio_mean_of_10_worst_long"),
        ("ers", "minimum_eqbal_ratio_mean_of_10_worst_short"),
        ("esl", "maximum_eqbal_ratio_std_long"),
        ("ess", "maximum_eqbal_ratio_std_short"),
        ("ct", "clip_threshold"),
    ]
    for k0, k1 in weights_keys:
        parser.add_argument(
            f"-{k0}",
            f"--{k1}",
            dest=k1,
            type=float,
            required=False,
            default=None,
            help=f"max {k1}",
        )
    parser.add_argument(
        "-i",
        "--index",
        dest="index",
        type=int,
        required=False,
        default=None,
        help="inspect particular config of given index",
    )
    parser.add_argument(
        "-d",
        "--dump_live_config",
        action="store_true",
        help="dump config",
    )

    args = parser.parse_args()

    opt_config = hjson.load(open("configs/optimize/default.hjson"))
    minsmaxs = {}
    for _, k1 in weights_keys:
        minsmaxs[k1] = opt_config[k1] if getattr(args, k1) is None else getattr(args, k1)
    klen = max([len(k) for k in minsmaxs])
    for k, v in minsmaxs.items():
        print(f"{k: <{klen}} {v}")

    with open(args.results_fpath) as f:
        results = [json.loads(x) for x in f.readlines()]
    print(f"{'n results': <{klen}} {len(results)}")
    passivbot_mode = determine_passivbot_mode(make_compatible(results[-1]["config"]))
    all_scores = []
    symbols = [s for s in results[0]["results"] if s != "config_no"]
    starting_balance = results[-1]["results"][symbols[0]]["starting_balance"]
    print(f"{'starting_balance': <{klen}} {starting_balance}")
    sides = ["long", "short"]
    for r in results:
        cfg = r["config"].copy()
        cfg.update(minsmaxs)
        ress = r["results"]
        all_scores.append({})
        scores_res = calc_scores(cfg, {s: r["results"][s] for s in symbols})
        scores, individual_scores, keys = (
            scores_res["scores"],
            scores_res["individual_scores"],
            scores_res["keys"],
        )
        for side in sides:
            all_scores[-1][side] = {
                "config": cfg[side],
                "score": scores[side],
                "individual_scores": individual_scores[side],
                "symbols_to_include": scores_res["symbols_to_include"][side],
                "stats": {sym: {k: v for k, v in ress[sym].items() if side in k} for sym in symbols},
                "config_no": ress["config_no"],
                "n_days": {sym: ress[sym]["n_days"] for sym in symbols},
            }
    best_candidate = {}
    for side in sides:
        scoress = sorted([sc[side] for sc in all_scores], key=lambda x: x["score"])
        best_candidate[side] = scoress[0]
        if args.index is not None:
            best_candidate[side] = [elm for elm in scoress if elm["config_no"] == args.index][0]
    best_config = {side: best_candidate[side]["config"] for side in sides}
    best_config = {
        "long": best_candidate["long"]["config"],
        "short": best_candidate["short"]["config"],
    }
    table_filepath = f"{args.results_fpath.replace('all_results.txt', '')}table_best_config.txt"
    if os.path.exists(table_filepath):
        os.remove(table_filepath)
    for side in sides:
        row_headers = ["symbol"] + [k[0] for k in keys] + ["n_days", "score"]
        table = PrettyTable(row_headers)
        for rh in row_headers:
            table.align[rh] = "l"
        table.title = (
            f"{side} (config no. {best_candidate[side]['config_no']},"
            + f" score {round_dynamic(best_candidate[side]['score'], 15)})"
        )
        for sym in sorted(
            symbols,
            key=lambda x: best_candidate[side]["individual_scores"][x],
            reverse=True,
        ):
            xs = [best_candidate[side]["stats"][sym][f"{k[0]}_{side}"] for k in keys]
            table.add_row(
                [("-> " if sym in best_candidate[side]["symbols_to_include"] else "") + sym]
                + [round_dynamic(x, 4) if np.isfinite(x) else x for x in xs]
                + [round(best_candidate[side]["n_days"][sym], 2)]
                + [best_candidate[side]["individual_scores"][sym]]
            )
        means = [
            np.mean(
                [
                    best_candidate[side]["stats"][s_][f"{k[0]}_{side}"]
                    for s_ in best_candidate[side]["symbols_to_include"]
                ]
            )
            for k in keys
        ]
        ind_scores_mean = np.mean(
            [
                best_candidate[side]["individual_scores"][sym]
                for sym in best_candidate[side]["symbols_to_include"]
            ]
        )
        table.add_row(
            ["mean"]
            + [round_dynamic(m, 4) if np.isfinite(m) else m for m in means]
            + [round(np.mean(list(best_candidate[side]["n_days"].values())), 2)]
            + [ind_scores_mean]
        )
        with open(make_get_filepath(table_filepath), "a") as f:
            output = table.get_string(border=True, padding_width=1)
            print(output)
            f.write(re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", output) + "\n\n")
    live_config = candidate_to_live_config(best_config)
    if args.dump_live_config:
        lc_fpath = make_get_filepath(f"{args.results_fpath.replace('.txt', '_best_config.json')}")
        print(f"dump_live_config {lc_fpath}")
        dump_live_config(live_config, lc_fpath)
    print(config_pretty_str(live_config))


if __name__ == "__main__":
    main()
