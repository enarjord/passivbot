"""Binance Portfolio Margin for debt-free, USDT-only UM perpetual accounts."""
import asyncio
import math
from decimal import Decimal, InvalidOperation

from exchanges.binance import BinanceBot
from exchanges.ccxt_bot import format_exchange_config_response
from live.balance_composition import unavailable_balance_composition
from passivbot import logging
from passivbot_exceptions import FatalBotException
from utils import symbol_to_coin


def _number(row: dict, field: str) -> Decimal:
    value = row[field]
    if isinstance(value, bool):
        raise ValueError(f'Binance PM: invalid numeric {field}')
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f'Binance PM: invalid numeric {field}') from None
    if not parsed.is_finite():
        raise ValueError(f'Binance PM: non-finite {field}')
    return parsed


class BinancePortfolioBot(BinanceBot):
    """Use PAPI account surfaces while preserving Binance order and broker contracts.

    The supported account contains only USDT collateral and USDT-settled UM
    perpetuals, with no margin debt, interest, negative balance, or CM position.
    Every balance snapshot proves that domain before returning USDT wallet
    balance. Account equity is USD-valued, collateral-adjusted, and includes
    unrealized PnL; it must not enter Passivbot's wallet-balance input.
    Private order notifications use the existing REST polling path.
    """

    is_portfolio_margin = True

    def create_ccxt_sessions(self):
        if self.user_info.get('quote', 'USDT') != 'USDT':
            raise FatalBotException('Binance PM supports only USDT-quoted UM perpetuals')
        super().create_ccxt_sessions()
        self.cca.options['portfolioMargin'] = True
        if self.ccp is not None:
            self.ccp.options['portfolioMargin'] = True
        logging.info('%s: Portfolio Margin UM mode enabled (PAPI)', self.exchange)

    def can_watch_orders(self) -> bool:
        return False

    async def _do_fetch_positions(self) -> list:
        # The inherited normalizer consumes raw positionAmt/positionSide rows,
        # whereas CCXT fetch_positions_risk() returns unified position objects.
        rows = await self.cca.papiGetUmPositionRisk()
        if not isinstance(rows, list):
            raise ValueError('Binance PM: invalid UM position-risk response')
        for row in rows:
            if _number(row, 'positionAmt') != 0 and not str(row['symbol']).endswith('USDT'):
                raise FatalBotException('Binance PM supports only USDT-settled UM positions')
        return rows

    async def _do_fetch_balance(self) -> dict:
        balances, cm_positions = await asyncio.gather(
            self.cca.papiGetBalance(), self.cca.papiGetCmPositionRisk()
        )
        return {'balances': balances, 'cm_positions': cm_positions}

    def _get_balance(self, fetched: dict) -> float:
        if self.quote != 'USDT':
            raise FatalBotException('Binance PM supports only USDT collateral')
        cm_positions = fetched['cm_positions']
        if not isinstance(cm_positions, list):
            raise ValueError('Binance PM: invalid CM position-risk response')
        for position in cm_positions:
            if _number(position, 'positionAmt') != 0:
                raise FatalBotException('Binance PM does not support CM position exposure')
        rows = fetched['balances']
        if not isinstance(rows, list):
            raise ValueError('Binance PM: invalid balance response')
        seen = set()
        wallet = None
        fields = ('totalWalletBalance', 'crossMarginAsset', 'crossMarginFree',
                  'crossMarginLocked', 'crossMarginBorrowed', 'crossMarginInterest',
                  'umWalletBalance', 'umUnrealizedPNL', 'cmWalletBalance', 'negativeBalance')
        for row in rows:
            asset = row['asset']
            if not isinstance(asset, str) or not asset or asset in seen:
                raise ValueError('Binance PM: invalid or duplicate balance asset')
            seen.add(asset)
            values = {field: _number(row, field) for field in fields}
            # Binance documents an empty CM PnL string for an unused CM ledger.
            # Accept it only after the independent all-CM-position absence proof.
            cm_pnl = row['cmUnrealizedPNL']
            if cm_pnl != '':
                cm_pnl = _number(row, 'cmUnrealizedPNL')
                if cm_pnl != 0:
                    raise FatalBotException('Binance PM does not support CM unrealized PnL')
            for field in ('crossMarginBorrowed', 'crossMarginInterest', 'negativeBalance', 'cmWalletBalance'):
                if values[field] != 0:
                    raise FatalBotException(f'Binance PM does not support nonzero {field}')
            if asset != 'USDT':
                if any(value != 0 for value in values.values()):
                    raise FatalBotException('Binance PM supports only USDT collateral and UM settlement')
                continue
            if any(values[field] < 0 for field in ('totalWalletBalance', 'crossMarginAsset',
                                                   'crossMarginFree', 'crossMarginLocked', 'umWalletBalance')):
                raise ValueError('Binance PM: negative wallet component')
            if values['crossMarginAsset'] != values['crossMarginFree'] + values['crossMarginLocked']:
                raise ValueError('Binance PM: inconsistent cross-margin wallet components')
            if values['totalWalletBalance'] != values['crossMarginAsset'] + values['umWalletBalance']:
                raise ValueError('Binance PM: inconsistent total wallet balance')
            wallet = values['totalWalletBalance']
        if wallet is None:
            raise ValueError('Binance PM: missing USDT wallet balance')
        normalized = float(wallet)
        if not math.isfinite(normalized) or (normalized == 0.0 and wallet != 0):
            raise ValueError('Binance PM: wallet balance exceeds numeric range')
        return normalized

    def _normalize_balance_diagnostics(self, fetched: object) -> dict:
        # The raw PAPI snapshot is not a CCXT unified balance map.
        return unavailable_balance_composition(reason='binance_portfolio_raw_balance')

    async def update_exchange_config_by_symbols(self, symbols):
        # PM has no per-symbol isolated/cross toggle. Failures setting required
        # leverage propagate to the existing exchange-configuration retry policy.
        symbols = list(symbols)
        results = await asyncio.gather(*(
            self.cca.set_leverage(self._calc_leverage_for_symbol(symbol), symbol=symbol)
            for symbol in symbols
        ))
        for symbol, result in zip(symbols, results):
            logging.info('%s: leverage=%s', symbol_to_coin(symbol, verbose=False) or symbol,
                         format_exchange_config_response(result))

    async def _prepare_protective_account(self):
        # CCXT fetch_position_mode() reads the ordinary FAPI wallet, not PAPI.
        response = await self.cca.papiGetUmPositionSideDual()
        dual_side = response['dualSidePosition']
        if dual_side is not True and str(dual_side).lower() != 'true':
            raise RuntimeError('Binance PM protective startup requires existing hedge position mode')
