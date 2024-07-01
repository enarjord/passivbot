import asyncio
import json
import passivbot_rust as pbr
import numpy as np
import pandas as pd
from procedures import utc_ms, make_get_filepath
from downloader import prepare_hlcs_forager
from pure_funcs import (
    extract_and_sort_by_keys_recursive,
    tuplify,
    get_template_live_config,
    ts_to_date_utc,
    denumpyize,
    process_forager_fills,
)
from njit_multisymbol import calc_noisiness_argsort_indices


def calc_recursive_entries_long(
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    balance,
    order_book_bid,
    ema_bands_lower,
    entry_grid_double_down_factor,
    entry_grid_spacing_weight,
    entry_grid_spacing_pct,
    entry_initial_ema_dist,
    entry_initial_qty_pct,
    wallet_exposure_limit,
    position_size,
    position_price,
    whole_grid=False,
):
    entries = []
    position_size_ = position_size
    position_price_ = position_price
    order_book_bid_ = order_book_bid
    i = 0
    infinite_loop_break = 30
    while True:
        i += 1
        if i > infinite_loop_break:
            break
        entry_qty, entry_price, entry_type = pbr.calc_grid_entry_long_py(
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
            balance,
            order_book_bid_,
            ema_bands_lower,
            entry_grid_double_down_factor,
            entry_grid_spacing_weight,
            entry_grid_spacing_pct,
            entry_initial_ema_dist,
            entry_initial_qty_pct,
            wallet_exposure_limit,
            position_size_,
            position_price_,
        )
        if entry_qty == 0.0:
            break
        if entries and entry_price == entries[-1][1]:
            break
        position_size_, position_price_ = pbr.calc_new_psize_pprice(
            position_size_, position_price_, entry_qty, entry_price, qty_step
        )
        order_book_bid_ = min(order_book_bid, entry_price)
        wallet_exposure = pbr.qty_to_cost(position_size_, position_price_, c_mult) / balance
        if "unstuck" in entry_type:
            if len(entries) == 0:
                # return unstucking entry only if it's the only one
                return [
                    (
                        entry_qty,
                        entry_price,
                        entry_type,
                        position_size_,
                        position_price_,
                        wallet_exposure,
                    )
                ]
        else:
            entries.append(
                (entry_qty, entry_price, entry_type, position_size_, position_price_, wallet_exposure)
            )
        if not whole_grid and position_size == 0.0:
            break
    return entries


async def main2():
    qty_step = 0.001
    price_step = 0.001
    min_qty = 0.001
    min_cost = 10.0
    c_mult = 1.0
    balance = 100000.0
    order_book_bid = 0.25
    ema_bands_lower = 0.25
    entry_grid_double_down_factor = 1.0876886495388454
    entry_grid_spacing_weight = 2.351196129873328
    entry_grid_spacing_pct = 0.05764972776535436
    entry_initial_ema_dist = 0.0
    entry_initial_qty_pct = 0.011934264419618748
    wallet_exposure_limit = 1.0
    position_size = 0.0
    position_price = 0.0

    res = calc_recursive_entries_long(
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        balance,
        order_book_bid,
        ema_bands_lower,
        entry_grid_double_down_factor,
        entry_grid_spacing_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        wallet_exposure_limit,
        position_size,
        position_price,
        whole_grid=True,
    )
    edf = pd.DataFrame(res, columns=["qty", "price", "type", "psize", "pprice", "wallet_exposure"])
    edf.loc[:, "pprice_diff"] = 1.0 - edf.price / edf.pprice.shift()
    grid_span = 1.0 - edf.iloc[-1].price / edf.iloc[0].price
    print("grid_span", grid_span)
    print(edf)


async def main():
    symbols = "BAKEUSDT,TONUSDT,YGGUSDT,PENDLEUSDT,RLCUSDT,IDEXUSDT,SSVUSDT,USTCUSDT,TAOUSDT,OMNIUSDT,SUPERUSDT,MDTUSDT,MAVIAUSDT,GMXUSDT,JASMYUSDT,AIUSDT,POLYXUSDT,NFPUSDT,ONDOUSDT,CKBUSDT,JUPUSDT,LEVERUSDT,ZETAUSDT,ENAUSDT,VANRYUSDT,MOVRUSDT,FTMUSDT,ORDIUSDT,RNDRUSDT,HIGHUSDT,TRUUSDT,PORTALUSDT,RSRUSDT,CHRUSDT,ARKMUSDT,SAGAUSDT,TOKENUSDT,1000SATSUSDT,1000BONKUSDT,WLDUSDT,XVGUSDT,ARUSDT,REZUSDT,JTOUSDT,1000FLOKIUSDT,AEVOUSDT,TNSRUSDT,WIFUSDT,PHBUSDT,BOMEUSDT,LPTUSDT,1000PEPEUSDT,UMAUSDT,FRONTUSDT,1000RATSUSDT,TRBUSDT,MYROUSDT,PEOPLEUSDT,BBUSDT,NOTUSDT"
    symbols += ",BTCUSDT,ETHUSDT,XRPUSDT,LINKUSDT"
    symbols = symbols.split(",")[-16:]
    symbols = sorted(set(symbols))

    print(symbols)
    try:
        mss = fetch_market_specific_settings_multi()
        json.dump(mss, open("tmp/mss_binance.json", "w"), indent=4)
    except:
        mss = json.load(open("tmp/mss_binance.json"))
    start_date = "2020-05-01"
    end_date = "2024-06-20"
    timestamps, hlcs = await prepare_hlcs_forager(symbols, start_date, end_date)

    noisiness_indices = calc_noisiness_argsort_indices(hlcs).astype(np.int32)
    print("noisiness_indices shape", noisiness_indices.shape)

    bot_params = {
        "long": {
            "close_grid_markup_range": 0.005,
            "close_grid_min_markup": 0.005,
            "close_grid_qty_pct": 0.1,
            "close_trailing_retracement_pct": 0.003,
            "close_trailing_grid_ratio": 0.5,
            "close_trailing_threshold_pct": 0.006,
            "entry_grid_double_down_factor": 1.0,
            "entry_grid_spacing_weight": 0.1,
            "entry_grid_spacing_pct": 0.05,
            "entry_initial_ema_dist": 0.001,
            "entry_initial_qty_pct": 0.005,
            "entry_trailing_retracement_pct": 0.03,
            "entry_trailing_grid_ratio": -0.5,
            "entry_trailing_threshold_pct": 0.08,
            "ema_span_0": 500.0,
            "ema_span_1": 1400.0,
            "n_positions": 6,
            "total_wallet_exposure_limit": 1.0,
            "unstuck_close_pct": 0.01,
            "unstuck_ema_dist": 0.0,
            "unstuck_loss_allowance_pct": 0.01,
            "unstuck_threshold": 0.6,
        },
        "short": {
            "close_grid_markup_range": 0.005,
            "close_grid_min_markup": 0.005,
            "close_grid_qty_pct": 0.1,
            "close_trailing_retracement_pct": 0.003,
            "close_trailing_grid_ratio": 0.5,
            "close_trailing_threshold_pct": 0.006,
            "entry_grid_double_down_factor": 1.0,
            "entry_grid_spacing_weight": 0.1,
            "entry_grid_spacing_pct": 0.05,
            "entry_initial_ema_dist": 0.001,
            "entry_initial_qty_pct": 0.005,
            "entry_trailing_retracement_pct": 0.03,
            "entry_trailing_grid_ratio": -0.5,
            "entry_trailing_threshold_pct": 0.08,
            "ema_span_0": 500.0,
            "ema_span_1": 1400.0,
            "n_positions": 6,
            "total_wallet_exposure_limit": 0.5,
            "unstuck_close_pct": 0.01,
            "unstuck_ema_dist": 0.0,
            "unstuck_loss_allowance_pct": 0.01,
            "unstuck_threshold": 0.6,
        },
    }

    bot_params["long"]["wallet_exposure_limit"] = (
        bot_params["long"]["total_wallet_exposure_limit"] / bot_params["long"]["n_positions"]
    )

    bot_params["short"]["wallet_exposure_limit"] = (
        bot_params["short"]["total_wallet_exposure_limit"] / bot_params["short"]["n_positions"]
    )

    starting_balance = 100000.0

    exchange_params = [
        {k: mss[symbol][k] for k in ["qty_step", "price_step", "min_qty", "min_cost", "c_mult"]}
        for symbol in symbols
    ]
    backtest_params = {"starting_balance": 100000.0, "maker_fee": 0.0002, "symbols": symbols}
    sts = utc_ms()
    fills = pbr.run_backtest(hlcs, noisiness_indices, bot_params, exchange_params, backtest_params)
    fdf = process_forager_fills(fills)
    print(fdf)
    print(f"duration seconds {(utc_ms() - sts) / 1000}")
    now_sec = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
    base_path = f"tmp/tests/{now_sec}/"
    fdf.to_csv(make_get_filepath(f"{base_path}rust_test_fills.csv"))
    np.save(make_get_filepath(f"{base_path}rust_test_hlcs.npy"), hlcs)
    json.dump(
        {"symbols": symbols, "start_date": start_date, "end_date": end_date},
        open(f"{base_path}args.json", "w"),
    )
    with open(f"{base_path}fills.txt", "w") as f:
        for fill in fills:
            f.write(json.dumps(denumpyize(fill)) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
