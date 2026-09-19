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
        ("close", "buy", "hedge_mode", ("buy", "short")),
        ("close", "sell", "hedge_mode", ("sell", "long")),
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
        ("unknown", "sell", "one_way_mode", ("sell", "short")),
    ],
)
def test_deduce_side_pside(trade_side, side, pos_mode, expected):
    payload = _make_fill(tradeSide=trade_side, side=side, posMode=pos_mode)
    assert deduce_side_pside(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"tradeSide": "unknown", "side": "sell", "posMode": "hedge_mode"},
        {"tradeSide": "unknown", "side": "", "posMode": "one_way_mode"},
        {"tradeSide": "", "side": "buy", "posMode": ""},
    ],
)
def test_deduce_side_pside_rejects_ambiguous_payload(payload):
    with pytest.raises(ValueError, match="cannot infer Bitget fill side/position_side"):
        deduce_side_pside(_make_fill(**payload))


def test_deduce_side_pside_uses_explicit_position_side():
    payload = _make_fill(tradeSide="close", side="", posMode="hedge_mode", posSide="short")
    assert deduce_side_pside(payload) == ("buy", "short")


@pytest.mark.parametrize('uta', [False, True])
@pytest.mark.parametrize('side,reduce_only,expected', [('buy',False,'long'), ('sell',False,'short'),
                                                      ('buy',True,'short'), ('sell',True,'long')])
def test_native_one_way_resting_orders_parse_with_hedge_runtime_default(uta, side, reduce_only, expected):
    from exchanges.bitget import BitgetBot
    bot = BitgetBot.__new__(BitgetBot)
    bot.is_uta = uta
    bot.hedge_mode = True
    order = {'id': '1', 'symbol': 'BTC/USDT:USDT', 'side': side, 'amount': 1.0,
             'timestamp': 1, 'reduceOnly': reduce_only,
             'info': {'posSide': 'net', 'side': side,
                      'tradeSide': 'close' if reduce_only else 'open',
                      'reduceOnly': 'YES' if reduce_only else 'NO'}}
    normalized = bot._normalize_open_orders([order])
    assert normalized[0]['position_side'] == expected
    assert bot._canonical_open_order_reduce_only(normalized[0]) is reduce_only
    assert bot.hedge_mode is True


def test_bitget_net_order_rejects_contradictory_hedge_mode():
    from exchanges.bitget import BitgetBot
    bot = BitgetBot.__new__(BitgetBot)
    bot.hedge_mode = True
    with pytest.raises(ValueError, match='contradictory'):
        bot._get_position_side_for_order({'info': {'posSide': 'net', 'holdMode': 'hedge_mode'}})
