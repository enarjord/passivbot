import pytest

from exchanges.bitget import deduce_side_pside


def _make_fill(**kwargs):
    base = {
        "tradeId": "0",
        "symbol": "X",
        "orderId": "0",
        "price": "0",
        "baseVolume": "0",
        "feeDetail": [],
        "side": "buy",
        "quoteVolume": "0",
        "profit": "0",
        "enterPointSource": "api",
        "tradeSide": "open",
        "posMode": "hedge_mode",
        "tradeScope": "taker",
        "marginCoin": "USDT",
        "cTime": "0",
    }
    base.update(kwargs)
    return base


@pytest.mark.parametrize(
    "trade_side, side, pos_mode, expected",
    [
        ("open", "buy", "hedge_mode", ("buy", "long")),
        ("close", "buy", "hedge_mode", ("sell", "long")),
        ("close", "sell", "hedge_mode", ("buy", "long")),
        ("reduce_close_long", "sell", "hedge_mode", ("sell", "long")),
        ("reduce_close_short", "buy", "hedge_mode", ("buy", "short")),
        ("burst_close_long", "sell", "hedge_mode", ("sell", "long")),
        ("burst_close_short", "buy", "hedge_mode", ("buy", "short")),
        ("offset_close_long", "sell", "hedge_mode", ("sell", "long")),
        ("offset_close_short", "buy", "hedge_mode", ("buy", "short")),
        ("delivery_close_long", "sell", "hedge_mode", ("sell", "long")),
        ("delivery_close_short", "buy", "hedge_mode", ("buy", "short")),
        ("dte_sys_adl_close_long", "sell", "hedge_mode", ("sell", "long")),
        ("dte_sys_adl_close_short", "buy", "hedge_mode", ("buy", "short")),
        ("buy_single", "buy", "one_way_mode", ("buy", "long")),
        ("sell_single", "sell", "one_way_mode", ("sell", "short")),
        ("reduce_buy_single", "buy", "one_way_mode", ("buy", "long")),
        ("reduce_sell_single", "sell", "one_way_mode", ("sell", "short")),
        ("burst_buy_single", "buy", "one_way_mode", ("buy", "long")),
        ("burst_sell_single", "sell", "one_way_mode", ("sell", "short")),
        ("delivery_buy_single", "buy", "one_way_mode", ("buy", "long")),
        ("delivery_sell_single", "sell", "one_way_mode", ("sell", "short")),
        ("dte_sys_adl_buy_in_single_side_mode", "buy", "one_way_mode", ("buy", "long")),
        ("dte_sys_adl_sell_in_single_side_mode", "sell", "one_way_mode", ("sell", "short")),
        ("unknown", "sell", "hedge_mode", ("sell", "long")),
        ("unknown", "", "one_way_mode", ("buy", "long")),
    ],
)
def test_deduce_side_pside(trade_side, side, pos_mode, expected):
    payload = _make_fill(tradeSide=trade_side, side=side, posMode=pos_mode)
    assert deduce_side_pside(payload) == expected
