import os
os.environ["NOJIT"] = "true"
import sys
import argparse
from prettytable import PrettyTable
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pure_funcs import ts_to_date_utc





def main():
    metric_explanations = {
        "adg_w_per_exp": "average daily gain, recent data weighted heavier",
        "adg_per_exp": "average daily gain for whole backtest",
        "exp_rts_mean": "exposure ratios mean: average wallet exposure during backtest",
        "time_at_max_exp": "ratio of hours spent at >90% exposure to hours spent at <90% exposure",
        "hrs_stuck_max": "how many hours in a stretch with no fills",
        "pa_dist_mean": "price action distance mean: the average distance between position price and market price",
        "pa_dist_std": "price action distance std: the standard deviation of the distance between position price and market price",
        "pa_dist_1pct_worst_mean": "mean of 1% highest pa_dist_mean",
        "loss_profit_rt": "loss to profit ratio: abs(sum(losses)) / sum(profit)",
        "n_days": "backtest number of days",
        "score": "used internally by optimizer",
    }
    parser = argparse.ArgumentParser(prog="sort backtest results", description="pretty view")
    parser.add_argument("results_fpath", type=str, help="path to results file")
    args = parser.parse_args()
    filenames = os.listdir(args.results_fpath)
    sides = ["long", "short"]
    results = []
    table_fpath = os.path.join(args.results_fpath, 'table_dump.txt')
    with open(table_fpath, 'w') as f:
        f.write('')
    for fname in filenames:
        try:
            if fname.endswith(".txt"):
                symbol = fname[: fname.find("_")]
                with open(os.path.join(args.results_fpath, fname)) as f:
                    content = f.read()
                #print(content)
                results.append({})
                for side, ih, iv in [('long', 3, 5), ('short', 12, 14)]:
                    headers = content.splitlines()[ih].replace('|', '').replace('->', '').split()
                    vals = content.splitlines()[iv].replace('|', '').replace('->', '').split()
                    items = [(x,y) for x, y in zip(headers, vals)]
                    results[-1][side] = {}
                    for x, y in items:
                        try:
                            results[-1][side][x] = float(y)
                        except:
                            results[-1][side][x] = y
        except Exception as e:
            #print(fname, e)
            pass
    with open(table_fpath, 'a') as f:
        max_len = max([len(k) for k in metric_explanations])
        today = ts_to_date_utc(time.time())[:10]
        line = f"### single symbol backtest metrics ###\n\nupdate: {today}\n\n"
        line += "note: symbols younger than 2021-05-01 are not optimized on their own,\n"
        line += "instead the best config from among the older symbols is chosen\n\n"
        line += "metric abbreviation comments:"
        f.write(line + '\n')
        print(line)
        for k, v in metric_explanations.items():
            line = f"{k: <{max_len}} {v}"
            f.write(line + '\n')
            print(line)
    for side in sides:
        sresults = sorted([x[side] for x in results if x], key=lambda x: x['score'], reverse=False)
        row_headers = list(sresults[0].keys())
        table = PrettyTable(row_headers)
        for rh in row_headers:
            table.align[rh] = "l"
        table.title = (f"{side}")
        for elm in sresults:
            table.add_row(list(elm.values()))
        print(table)
        with open(table_fpath, 'a') as f:
            f.write(table.get_string(border=True, padding_width=1) + '\n\n')


if __name__ == "__main__":
    main()


