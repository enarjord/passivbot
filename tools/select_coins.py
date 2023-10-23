import pandas as pd
import sys
import pprint
import argparse
import os


def floatify(xs):
    if isinstance(xs, (int, float)):
        return float(xs)
    elif isinstance(xs, str):
        try:
            return float(xs)
        except ValueError:
            return xs
    elif isinstance(xs, bool):
        return xs
    elif isinstance(xs, list):
        return [floatify(x) for x in xs]
    elif isinstance(xs, tuple):
        return tuple(floatify(x) for x in xs)
    elif isinstance(xs, dict):
        return {k: floatify(v) for k, v in xs.items()}
    else:
        return xs


def main():
    parser = argparse.ArgumentParser(
        prog="select_coins", description="select coins from table dump"
    )
    parser.add_argument(
        "path_to_table_dump", type=str, help="path to live config to test"
    )
    parser.add_argument(
        "-adg",
        type=float,
        required=False,
        dest="adg",
        default=0.001,
        help="min adg",
    )
    parser.add_argument(
        "-nd",
        type=float,
        required=False,
        dest="min_n_days",
        default=600.0,
        help="min n days",
    )
    parser.add_argument(
        "-ns",
        type=int,
        required=False,
        dest="n_syms",
        default=8,
        help="n syms",
    )
    args = parser.parse_args()

    path_to_table_dump = (
        args.path_to_table_dump
        if args.path_to_table_dump.endswith("table_dump.txt")
        else os.path.join(args.path_to_table_dump, "table_dump.txt")
    )
    with open(args.path_to_table_dump) as f:
        all_lines = f.readlines()
    lines = [line for line in all_lines if "USDT" in line]
    longlines = [line for line in lines[: len(lines) // 2]]
    longlines = floatify([line.replace("|", "").split() for line in longlines])
    shortlines = [line for line in lines[len(lines) // 2 :]]
    shortlines = floatify([line.replace("|", "").split() for line in shortlines])
    line_with_headers = [line for line in all_lines if "| symbol" in line][0]
    column_names = line_with_headers.replace("|", "").split()
    longs = pd.DataFrame(longlines, columns=column_names)
    shorts = pd.DataFrame(shortlines, columns=column_names)
    longsf = longs[longs["adg_w_per_exp"] >= args.adg]
    longsf = longsf[longsf["adg_per_exp"] >= args.adg]
    shortsf = shorts[shorts["adg_w_per_exp"] >= args.adg]
    shortsf = shortsf[shortsf["adg_per_exp"] >= args.adg]
    longsf = longsf[longsf["n_days"] >= args.min_n_days]
    shortsf = shortsf[shortsf["n_days"] >= args.min_n_days]

    syms = set(longsf.symbol) & set(shortsf.symbol)
    nscores = {}
    for sym in syms:
        longr = longs[longs.symbol == sym]
        shortr = shorts[shorts.symbol == sym]
        nscores[sym] = (
            longr.adg_per_exp.iloc[0]
            + longr.adg_w_per_exp.iloc[0]
            + shortr.adg_per_exp.iloc[0]
            + shortr.adg_w_per_exp.iloc[0]
        ) / 4

    syms_sorted_by_nscores = sorted(nscores.items(), key=lambda x: x[1], reverse=True)
    syms = [
        s for s in syms if s in [x[0] for x in syms_sorted_by_nscores][: args.n_syms]
    ]
    print("long")
    print(longs[longs.symbol.isin(syms)].sort_values("adg_per_exp", ascending=False))
    print("\nshort")
    print(shorts[shorts.symbol.isin(syms)].sort_values("adg_per_exp", ascending=False))

    print("long+short scores")
    max_len = max([len(k) for k in nscores])
    for k, v in sorted(nscores.items(), key=lambda x: x[1], reverse=True):
        print(f"{k: <{max_len}} {v:.6f}")
    print(f"n syms adg > {args.adg}: {len(nscores)}")

    print("selected syms", sorted(syms))
    print("n syms", len(syms))


if __name__ == "__main__":
    main()
