"""Synthetic PM contracts; no network, credentials, or account operations."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from exchanges.binance import BinanceBot
from exchanges.binance_portfolio import BinancePortfolioBot
from fill_events_manager import _build_fetcher_for_bot
from passivbot_exceptions import FatalBotException


def balance_row(asset='USDT', **overrides):
    row = dict(asset=asset, totalWalletBalance='1000', crossMarginAsset='700',
               crossMarginFree='700', crossMarginLocked='0', crossMarginBorrowed='0',
               crossMarginInterest='0', umWalletBalance='300', umUnrealizedPNL='50',
               cmWalletBalance='0', cmUnrealizedPNL='', negativeBalance='0')
    row.update(overrides)
    return row


def pm_bot():
    bot = BinancePortfolioBot.__new__(BinancePortfolioBot)
    bot.exchange = 'binance'
    bot.quote = 'USDT'
    bot.markets_dict = {}
    bot.get_symbol_id_inv = lambda value: {'BTCUSDT': 'BTC/USDT:USDT'}[value]
    bot.coin_to_symbol = lambda value, verbose=False: value
    bot._shutdown_requested = lambda: False
    bot.cca = SimpleNamespace(
        papiGetAccount=AsyncMock(return_value={'accountEquity': '1050'}),
        papiGetBalance=AsyncMock(return_value=[balance_row()]),
        papiGetCmPositionRisk=AsyncMock(return_value=[]),
    )
    return bot


@pytest.mark.asyncio
async def test_pm_wallet_excludes_upnl_before_consumer_adds_it():
    bot = pm_bot()
    wallet = await bot.fetch_balance()
    assert wallet == 1000.0
    assert wallet + 50.0 == 1050.0
    bot.cca.papiGetAccount.assert_not_awaited()


@pytest.mark.asyncio
async def test_pm_position_hook_returns_raw_rows_for_inherited_normalizer():
    bot = pm_bot()
    raw = dict(symbol='BTCUSDT', positionAmt='-0.01', entryPrice='50000',
               positionSide='SHORT', updateTime=1_700_000_000_000)
    unified = dict(symbol='BTC/USDT:USDT', side='short', contracts=0.01,
                   entryPrice=50000.0, info=deepcopy(raw))
    bot.cca.fetch_positions_risk = AsyncMock(return_value=[unified])
    bot.cca.papiGetUmPositionRisk = AsyncMock(return_value=[raw])
    positions = await bot.fetch_positions()
    assert positions[0]['symbol'] == 'BTC/USDT:USDT'
    assert positions[0]['size'] == -0.01
    assert positions[0]['position_side'] == 'short'
    assert positions[0]['price'] == 50000.0
    bot.cca.fetch_positions_risk.assert_not_awaited()
    bot.cca.papiGetUmPositionRisk.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize('portfolio_margin', [False, True])
@pytest.mark.parametrize('bounded', [False, True])
async def test_binance_fill_factory_routes_income_to_actual_account(portfolio_margin, bounded):
    bot = pm_bot() if portfolio_margin else BinanceBot.__new__(BinanceBot)
    bot.exchange = 'binance'
    bot._shutdown_requested = lambda: False
    bot.cca = SimpleNamespace(
        fapiprivate_get_income=AsyncMock(return_value=[]),
        papiGetUmIncome=AsyncMock(return_value=[]),
    )
    fetcher = _build_fetcher_for_bot(bot, [])
    assert await fetcher._fetch_income_by_type(
        'REALIZED_PNL', 0 if bounded else None, 60000 if bounded else None
    ) == []
    expected = bot.cca.papiGetUmIncome if portfolio_margin else bot.cca.fapiprivate_get_income
    other = bot.cca.fapiprivate_get_income if portfolio_margin else bot.cca.papiGetUmIncome
    params = {'incomeType': 'REALIZED_PNL', 'limit': 1000}
    if bounded:
        params.update(startTime=0, endTime=60000)
    expected.assert_awaited_once_with(params=params)
    other.assert_not_awaited()


def payload(rows=None, cm_positions=None):
    return {'balances': [balance_row()] if rows is None else rows,
            'cm_positions': [] if cm_positions is None else cm_positions}


@pytest.mark.parametrize('field', [
    'totalWalletBalance', 'crossMarginAsset', 'crossMarginFree', 'crossMarginLocked',
    'crossMarginBorrowed', 'crossMarginInterest', 'umWalletBalance', 'umUnrealizedPNL',
    'cmWalletBalance', 'cmUnrealizedPNL', 'negativeBalance',
])
@pytest.mark.parametrize('invalid', [None, True, 'NaN', 'Infinity'])
def test_pm_rejects_invalid_required_wallet_facts(field, invalid):
    with pytest.raises((KeyError, ValueError)):
        pm_bot()._get_balance(payload([balance_row(**{field: invalid})]))


@pytest.mark.parametrize('field', [
    'totalWalletBalance', 'crossMarginAsset', 'crossMarginFree', 'crossMarginLocked',
    'crossMarginBorrowed', 'crossMarginInterest', 'umWalletBalance', 'umUnrealizedPNL',
    'cmWalletBalance', 'cmUnrealizedPNL', 'negativeBalance',
])
def test_pm_rejects_missing_required_wallet_facts(field):
    row = balance_row()
    del row[field]
    with pytest.raises(KeyError):
        pm_bot()._get_balance(payload([row]))


@pytest.mark.parametrize('field', ['crossMarginBorrowed', 'crossMarginInterest',
                                   'negativeBalance', 'cmWalletBalance', 'cmUnrealizedPNL'])
def test_pm_rejects_debt_and_other_settlement_ledgers(field):
    with pytest.raises(FatalBotException, match='does not support'):
        pm_bot()._get_balance(payload([balance_row(**{field: '0.01'})]))


def test_pm_rejects_other_collateral_even_when_it_is_not_allocated_to_futures():
    other = balance_row('BTC', totalWalletBalance='0.1', crossMarginAsset='0.1',
                        crossMarginFree='0.1', umWalletBalance='0', umUnrealizedPNL='0')
    with pytest.raises(FatalBotException, match='only USDT collateral'):
        pm_bot()._get_balance(payload([balance_row(), other]))


@pytest.mark.parametrize('positions,error', [
    (None, ValueError), ({}, ValueError), ([{}], KeyError),
    ([{'positionAmt': 'NaN'}], ValueError),
    ([{'positionAmt': '1', 'unRealizedProfit': '0'}], FatalBotException),
])
def test_pm_requires_independent_proof_of_no_cm_positions(positions, error):
    with pytest.raises(error):
        pm_bot()._get_balance({'balances': [balance_row()], 'cm_positions': positions})


def test_pm_accepts_documented_empty_cm_pnl_after_zero_position_proof():
    assert pm_bot()._get_balance(payload(cm_positions=[{'positionAmt': '0'}])) == 1000.0


@pytest.mark.parametrize('rows', [[], [balance_row(), balance_row()],
                                 [balance_row(totalWalletBalance='1050')],
                                 [balance_row(crossMarginAsset='800')]])
def test_pm_rejects_missing_duplicate_or_inconsistent_wallet(rows):
    with pytest.raises(ValueError):
        pm_bot()._get_balance(payload(rows))


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['papiGetBalance', 'papiGetCmPositionRisk'])
async def test_pm_balance_fetch_failures_propagate(method):
    bot = pm_bot()
    getattr(bot.cca, method).side_effect = OSError('synthetic transport failure')
    with pytest.raises(OSError):
        await bot.fetch_balance()


@pytest.mark.asyncio
async def test_pm_position_rejects_non_usdt_um_exposure():
    bot = pm_bot()
    bot.cca.papiGetUmPositionRisk = AsyncMock(return_value=[
        {'symbol': 'BTCUSDC', 'positionAmt': '0.01'}])
    with pytest.raises(FatalBotException, match='only USDT-settled'):
        await bot.fetch_positions()


@pytest.mark.asyncio
async def test_pm_leverage_failure_propagates_without_margin_mode_write():
    bot = pm_bot()
    bot._calc_leverage_for_symbol = lambda symbol: 3
    bot.cca.set_margin_mode = AsyncMock()
    bot.cca.set_leverage = AsyncMock(side_effect=RuntimeError('synthetic leverage failure'))
    with pytest.raises(RuntimeError, match='synthetic leverage failure'):
        await bot.update_exchange_config_by_symbols(['BTC/USDT:USDT'])
    bot.cca.set_margin_mode.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('dual', [True, 'true', 'TRUE'])
async def test_pm_protective_startup_reads_pm_hedge_mode(dual):
    bot = pm_bot()
    bot.cca.papiGetUmPositionSideDual = AsyncMock(return_value={'dualSidePosition': dual})
    bot.cca.fetch_position_mode = AsyncMock()
    await bot._prepare_protective_account()
    bot.cca.fetch_position_mode.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [{}, {'dualSidePosition': False}, {'dualSidePosition': None}])
async def test_pm_protective_startup_rejects_absent_or_one_way_mode(response):
    bot = pm_bot()
    bot.cca.papiGetUmPositionSideDual = AsyncMock(return_value=response)
    with pytest.raises((KeyError, RuntimeError)):
        await bot._prepare_protective_account()


@pytest.fixture(params=[False, True], ids=["rest-only", "with-pro-client"])
async def signed_pm_bot(request):
    """Run installed CCXT request construction with synthetic signing inputs only."""
    from urllib.parse import urlparse, parse_qs
    from procedures import load_broker_code

    bot = pm_bot()
    bot.user_info = {'apiKey': 'synthetic-key', 'secret': 'synthetic-secret', 'quote': 'USDT'}
    bot.exchange_ccxt_id = 'binanceusdm'
    bot.ws_enabled = request.param
    bot.endpoint_override = None
    bot.config = {"live": {"execution": {"time_in_force": "gtc"}}}
    bot.broker_code = load_broker_code('binance')
    bot.custom_id_max_length = 36
    bot.create_ccxt_sessions()
    client = bot.cca
    client.set_markets([{
        'id': 'BTCUSDT', 'symbol': 'BTC/USDT:USDT', 'base': 'BTC', 'quote': 'USDT',
        'settle': 'USDT', 'baseId': 'BTC', 'quoteId': 'USDT', 'settleId': 'USDT',
        'type': 'swap', 'spot': False, 'margin': False, 'swap': True, 'future': False,
        'option': False, 'contract': True, 'linear': True, 'inverse': False, 'active': True,
        'precision': {'amount': 0.001, 'price': 0.1},
        'limits': {'amount': {'min': 0.001}, 'price': {'min': 0.1}},
        'info': {'orderTypes': ['LIMIT', 'MARKET']},
    }])
    requests = []
    raw_order = dict(symbol='BTCUSDT', orderId=123, clientOrderId='synthetic',
                     origQty='0.001', executedQty='0', price='50000',
                     side='BUY', positionSide='LONG', type='LIMIT', status='NEW',
                     timeInForce='GTC', updateTime=1_700_000_000_000)

    async def capture(url, method='GET', headers=None, body=None):
        parsed = urlparse(url)
        params = parse_qs(body or parsed.query)
        requests.append((parsed.hostname, parsed.path, method, headers, params))
        assert headers['X-MBX-APIKEY'] == 'synthetic-key'
        assert 'signature' in params
        if parsed.path.endswith('/balance'):
            return [balance_row()]
        if parsed.path.endswith('/positionRisk'):
            return []
        if parsed.path.endswith('/positionSide/dual'):
            return {'dualSidePosition': True} if method == 'GET' else {'code': 200}
        if parsed.path.endswith('/income') or parsed.path.endswith('/userTrades'):
            return []
        if parsed.path.endswith('/openOrders'):
            return []
        if parsed.path.endswith('/leverage'):
            return {'leverage': 3, 'symbol': 'BTCUSDT'}
        return deepcopy(raw_order)

    client.fetch = capture  # Replace the transport, after real CCXT signing.
    try:
        yield bot, requests
    finally:
        await client.close()
        if bot.ccp is not None:
            await bot.ccp.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('position_side,side', [('long', 'buy'), ('long', 'sell'),
                                                ('short', 'sell'), ('short', 'buy')])
@pytest.mark.parametrize('order_type', ['limit', 'market'])
async def test_pm_signed_orders_preserve_broker_hedge_and_execution_fields(
    signed_pm_bot, position_side, side, order_type
):
    bot, requests = signed_pm_bot
    custom_id = bot.format_custom_id_single(1)
    params = bot._build_order_params({'type': order_type, 'position_side': position_side,
                                      'custom_id': custom_id})
    await bot.cca.create_order('BTC/USDT:USDT', order_type, side, 0.001,
                               50000.0 if order_type == 'limit' else None, params)
    host, path, method, headers, request = requests[-1]
    assert host == 'papi.binance.com'
    assert path == '/papi/v1/um/order' and method == 'POST'
    assert request['symbol'] == ['BTCUSDT']
    assert request['positionSide'] == [position_side.upper()]
    assert request['side'] == [side.upper()]
    assert request['newClientOrderId'] == [custom_id]
    assert custom_id.startswith('x-' + bot.broker_code + '0x')
    assert len(custom_id) <= 36
    assert 'reduceOnly' not in request
    assert request.get('timeInForce') == (['GTC'] if order_type == 'limit' else None)


@pytest.mark.asyncio
async def test_pm_signed_reads_config_and_cancellation_use_papi(signed_pm_bot):
    bot, requests = signed_pm_bot
    assert await bot.fetch_balance() == 1000.0
    assert await bot.fetch_positions() == []
    await bot._prepare_protective_account()
    await bot.update_exchange_config()
    bot._calc_leverage_for_symbol = lambda symbol: 3
    await bot.update_exchange_config_by_symbols(['BTC/USDT:USDT'])
    await bot.cca.cancel_order('123', 'BTC/USDT:USDT')
    await bot.cca.fetch_open_orders('BTC/USDT:USDT')
    await bot.cca.fetch_my_trades('BTC/USDT:USDT', limit=10)
    fetcher = _build_fetcher_for_bot(bot, [])
    await fetcher._fetch_income_by_type('REALIZED_PNL', None, None)
    assert {path for host, path, method, headers, params in requests} == {
        '/papi/v1/balance', '/papi/v1/cm/positionRisk', '/papi/v1/um/positionRisk',
        '/papi/v1/um/positionSide/dual', '/papi/v1/um/leverage', '/papi/v1/um/order',
        '/papi/v1/um/openOrders', '/papi/v1/um/userTrades', '/papi/v1/um/income',
    }
    assert all(host == 'papi.binance.com' for host, *_ in requests)
    assert bot.can_watch_orders() is False
    assert bot.cca.options['broker']['future'] == 'x-' + bot.broker_code
    if bot.ccp is not None:
        assert bot.ccp.options['portfolioMargin'] is True
        assert bot.ccp.options['broker']['future'] == 'x-' + bot.broker_code


@pytest.mark.parametrize('quote', ['USDC', 'BTC', '', None])
def test_pm_rejects_unsupported_quote_before_creating_sessions(quote):
    bot = pm_bot()
    bot.user_info = {'quote': quote}
    with pytest.raises(FatalBotException, match='only USDT-quoted'):
        bot.create_ccxt_sessions()


@pytest.mark.parametrize('amount', ['1e999', '1e-999'])
def test_pm_rejects_wallet_overflow(amount):
    with pytest.raises(ValueError, match='numeric range'):
        pm_bot()._get_balance(payload([balance_row(totalWalletBalance=amount,
            crossMarginAsset=amount, crossMarginFree=amount, umWalletBalance='0')]))


@pytest.mark.asyncio
async def test_pm_failed_balance_snapshot_recovers_without_hidden_state():
    bot = pm_bot()
    bot.cca.papiGetBalance.side_effect = [OSError('synthetic read failure'), [balance_row()]]
    with pytest.raises(OSError):
        await bot.fetch_balance()
    assert await bot.fetch_balance() == 1000.0


@pytest.mark.asyncio
async def test_pm_income_failure_propagates():
    bot = pm_bot()
    bot.cca.papiGetUmIncome = AsyncMock(side_effect=OSError('synthetic read failure'))
    fetcher = _build_fetcher_for_bot(bot, [])
    with pytest.raises(OSError):
        await fetcher._fetch_income_by_type('COMMISSION', 0, 60000)
