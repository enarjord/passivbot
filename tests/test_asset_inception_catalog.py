import pytest

from asset_inception_catalog import (
    asset_venue_bindings,
    is_non_crypto_instrument,
    resolve_asset_identity,
)


def market(base, native_id, info=None, *, quote="USDT"):
    return {
        "id": native_id,
        "base": base,
        "symbol": f"{base}/{quote}:{quote}",
        "quote": quote,
        "settle": quote,
        "swap": True,
        "info": info or {},
    }


def test_okx_equity_qnt_cannot_inherit_quant_token_history():
    equity = market("QNT", "QNT-USDT-SWAP", {"category": "1", "instCategory": "3"})
    assert is_non_crypto_instrument("okx", equity)
    assert resolve_asset_identity("okx", equity) is None

    legacy_snapshot = market("QNT", "QNT-USDT-SWAP", {"category": "1"})
    assert is_non_crypto_instrument("okx", legacy_snapshot)

    crypto = market("QUANT", "QUANT-USDT-SWAP", {"instCategory": "1"})
    quant = resolve_asset_identity("okx", crypto)
    assert not is_non_crypto_instrument("okx", crypto)
    assert quant.cmc_id == 3155
    assert ("okx", "QUANT-USDT-SWAP", "QUANT/USDT:USDT") in asset_venue_bindings(quant)
    assert ("okx", "QNT-USDT-SWAP", "QNT/USDT:USDT") not in asset_venue_bindings(quant)


@pytest.mark.parametrize(
    ("first", "second", "cmc_id"),
    [
        (("binanceusdm", "TON", "TONUSDT"), ("okx", "GRAM", "GRAM-USDT-SWAP"), 11419),
        (("okx", "MATIC", "MATIC-USDT-SWAP"), ("gate", "POL", "POL_USDT"), 28321),
        (("binance", "LEND", "LENDUSDT"), ("okx", "AAVE", "AAVE-USDT-SWAP"), 7278),
        (("okx", "RNDR", "RNDR-USDT-SWAP"), ("okx", "RENDER", "RENDER-USDT-SWAP"), 5690),
    ],
)
def test_verified_rename_and_migration_resolve_to_one_asset(first, second, cmc_id):
    identity = resolve_asset_identity(first[0], market(first[1], first[2]))
    assert identity.cmc_id == cmc_id
    assert identity == resolve_asset_identity(second[0], market(second[1], second[2]))


@pytest.mark.parametrize(
    ("exchange", "base", "native_id"),
    [
        ("okx", "APX", "APX-USDT-SWAP"),
        ("anything", "BTW", "BTWUSDT"),
        ("okx", "BTC", "NOT-BITCOIN-USDT-SWAP"),
        ("binance", "XBT", "BTCUSDT"),
    ],
)
def test_symbol_or_contract_id_alone_cannot_supply_asset_identity(exchange, base, native_id):
    assert resolve_asset_identity(exchange, market(base, native_id)) is None


def test_premarket_perpetual_cannot_supply_asset_identity():
    premarket = market("CC", "CC_USDT", {"is_pre_market": True})
    assert resolve_asset_identity("gateio", premarket) is None


def test_reused_okx_market_code_cannot_supply_previous_asset_identity():
    reviewed = market("BTW", "BTW-USDT-SWAP", {"instIdCode": 523548})
    assert resolve_asset_identity("okx", reviewed).cmc_id == 39158
    changed = market("BTW", "BTW-USDT-SWAP", {"instIdCode": 999000})
    assert resolve_asset_identity("okx", changed) is None
