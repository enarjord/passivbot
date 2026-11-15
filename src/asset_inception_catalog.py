"""Public instrument identities and vetted starting observations for universal coin age.

The provider's stable asset ID joins the first historical price to the actual
asset. Bind a new asset by reviewing the venue's instrument and its underlying.
The CoinPaprika asset-market directory can corroborate the underlying:
https://docs.coinpaprika.com/api-reference/coins/get-markets-by-coin-id

General instrument and price sources:
- https://www.okx.com/docs-v5/en/#public-data-rest-api-get-instruments
- https://github.com/binance/binance-public-data
- https://docs.coingecko.com/reference/derivatives-exchanges-id
- https://bybit-exchange.github.io/docs/v5/market/instrument
- https://www.weex.com/api-doc/contract/Market_API/GetContractInfo
- https://www.bitget.com/api-doc/contract/market/Get-All-Symbols-Contracts
- https://openapidoc.bitunix.com/doc/market/get_trading_pairs.html
- https://www.gate.com/docs/developers/apiv4/#list-all-futures-contracts
- https://www.kucoin.com/docs-new/rest/futures-trading/market-data/get-all-symbols
- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals
- https://coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency

Only verified replacement assets inherit a predecessor's price observations:
- TON to GRAM: https://ton.org/media/
- MATIC to POL: https://forum.polygon.technology/t/pip-46-matic-to-pol-migration/19672
- RNDR to RENDER: https://know.rendernetwork.com/basics/migrating-tokens
- LEND to AAVE: https://governance.aave.com/t/welcome-to-aaves-governance-discussion/7

Lifecycle bounds for rejecting pre-launch quotes and contract-only history:
- https://www.circle.com/fr/blog/cosmos-atoms-now-open-for-trading-and-redeemable-on-poloniex
- https://support.avax.network/en/articles/8460478-what-is-a-mainnet
- https://blog.kraken.com/news/bitcoin-cash-and-a-critical-alert-for-margin-traders-part-2-2
- https://www.okx.com/en-gb/help/okx-to-list-pre-market-perpetual-futures-for-cc-canton-network-crypto
- https://wiki.polkadot.com/general/faq/
- https://filecoin.io/blog/filecoin-mainnet-is-live
- https://morpho.org/blog/the-morpho-dao-to-enable-transferability-for-morpho/
- https://www.sui.io/token-schedule

Spot trade times are rounded up to milliseconds. Whole minute candles use
the end of the minute as a conservative time by which trading was observed.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping


@dataclass(frozen=True)
class AssetIdentity:
    name: str
    cmc_id: int
    cmc_symbol: str
    earliest_eligible_ms: int
    observation_ms: int | None = None
    observation_source: str | None = None


def _utc_timestamp_ms(value: str) -> int:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (timestamp - datetime(1970, 1, 1, tzinfo=timezone.utc)) // timedelta(milliseconds=1)


_CMC_MAP_DOCS = "https://coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency"
_KRAKEN_ATOM_TRADES = "https://api.kraken.com/0/public/Trades?pair=ATOMUSD&count=5&since=1555891200000000000"
_KRAKEN_BCH_TRADES = "https://api.kraken.com/0/public/Trades?pair=BCHUSD&count=5&since=1501545600000000000"
_KRAKEN_CC_TRADES = "https://api.kraken.com/0/public/Trades?pair=CCUSD&count=5&since=1762732800000000000"
_KRAKEN_DOT_TRADES = "https://api.kraken.com/0/public/Trades?pair=DOTUSD&count=5&since=1597708800000000000"
_BINANCE_AVAX_PRICES = "https://data.binance.vision/data/spot/monthly/klines/AVAXUSDT/1m/AVAXUSDT-1m-2020-09.zip"
_BINANCE_FIL_PRICES = "https://data.binance.vision/data/spot/monthly/klines/FILUSDT/1m/FILUSDT-1m-2020-10.zip"
_BINANCE_MATIC_PRICES = "https://data.binance.vision/data/spot/monthly/klines/MATICBTC/1m/MATICBTC-1m-2019-04.zip"
_BINANCE_LEND_PRICES = "https://data.binance.vision/data/spot/monthly/klines/LENDBTC/1m/LENDBTC-1m-2017-12.zip"


def _asset(
    symbol: str,
    name: str,
    cmc_id: int,
    observation: str,
    source: str = _CMC_MAP_DOCS,
    *,
    earliest: str | None = None,
) -> AssetIdentity:
    # Evidence before the reviewed observation needs a separately established
    # asset-availability bound. An opening candle bucket can predate trading.
    floor = earliest or observation
    return AssetIdentity(
        name,
        cmc_id,
        symbol,
        _utc_timestamp_ms(floor),
        _utc_timestamp_ms(observation),
        source,
    )


_ASSETS = {
    "AAVE": _asset("AAVE", "Aave (LEND/AAVE)", 7278, "2017-12-12T08:50:59.999Z", _BINANCE_LEND_PRICES),
    "ADA": _asset("ADA", "Cardano", 2010, "2017-10-01T20:30:00Z"),
    "ALGO": _asset("ALGO", "Algorand", 4030, "2019-06-21T00:10:00Z"),
    "APT": _asset("APT", "Aptos", 21794, "2022-10-19T02:00:00Z"),
    "ARB": _asset("ARB", "Arbitrum", 11841, "2023-03-23T13:10:00Z"),
    "ASTER": _asset("ASTER", "Aster", 36341, "2025-09-17T14:10:00Z"),
    "ATOM": _asset("ATOM", "Cosmos", 3794, "2019-04-22T18:13:37.208Z", _KRAKEN_ATOM_TRADES),
    "AVAX": _asset(
        "AVAX",
        "Avalanche",
        5805,
        "2020-09-22T06:30:59.999Z",
        _BINANCE_AVAX_PRICES,
        earliest="2020-09-21",
    ),
    "BCH": _asset("BCH", "Bitcoin Cash", 1831, "2017-08-01T16:01:37.148Z", _KRAKEN_BCH_TRADES),
    "BNB": _asset("BNB", "BNB", 1839, "2017-07-25T04:30:00Z"),
    "BTC": _asset("BTC", "Bitcoin", 1, "2010-07-13T00:05:00Z"),
    "BTW": _asset("BTW", "Bitway", 39158, "2026-03-02T08:05:00Z"),
    "CC": _asset("CC", "Canton", 37263, "2025-11-10T07:00:50.893Z", _KRAKEN_CC_TRADES),
    "CHZ": _asset("CHZ", "Chiliz", 4066, "2019-07-01T19:00:00Z"),
    "DOGE": _asset("DOGE", "Dogecoin", 74, "2013-12-15T14:40:00Z"),
    "DOT": _asset(
        "DOT",
        "Polkadot",
        6636,
        "2020-08-18T17:02:07.918Z",
        _KRAKEN_DOT_TRADES,
        earliest="2020-08-18T16:39:00Z",
    ),
    "ENA": _asset("ENA", "Ethena", 30171, "2024-04-02T07:25:00Z"),
    "ETC": _asset("ETC", "Ethereum Classic", 1321, "2016-07-24T13:40:00Z"),
    "ETH": _asset("ETH", "Ethereum", 1027, "2015-08-07T14:45:00Z"),
    "FIL": _asset(
        "FIL",
        "Filecoin",
        2280,
        "2020-10-15T17:00:59.999Z",
        _BINANCE_FIL_PRICES,
        earliest="2020-10-15T15:00:00Z",
    ),
    "GRAM": _asset("GRAM", "Gram (formerly Toncoin)", 11419, "2021-08-26T14:00:00Z"),
    "HBAR": _asset("HBAR", "Hedera", 4642, "2019-09-17T09:10:00Z"),
    "HYPE": _asset("HYPE", "Hyperliquid", 32196, "2024-11-29T07:45:00Z"),
    "ICP": _asset("ICP", "Internet Computer", 8916, "2021-05-10T11:00:00Z"),
    "INJ": _asset("INJ", "Injective", 7226, "2020-10-20T04:10:00Z"),
    "KAS": _asset("KAS", "Kaspa", 20396, "2022-06-01T09:35:00Z"),
    "LINK": _asset("LINK", "Chainlink", 1975, "2017-09-20T20:50:00Z"),
    "LTC": _asset("LTC", "Litecoin", 2, "2013-04-28T18:45:00Z"),
    "MORPHO": _asset("MORPHO", "Morpho", 34104, "2024-11-21T10:30:00Z"),
    "NEAR": _asset("NEAR", "NEAR Protocol", 6535, "2020-10-13T05:10:00Z"),
    "ONDO": _asset("ONDO", "Ondo", 21159, "2024-01-18T12:05:00Z"),
    "OP": _asset("OP", "Optimism", 11840, "2022-05-31T16:30:00Z"),
    "POL": _asset("POL", "Polygon (MATIC/POL)", 28321, "2019-04-26T15:00:59.999Z", _BINANCE_MATIC_PRICES),
    "QNT": _asset("QNT", "Quant", 3155, "2018-08-10T20:25:00Z"),
    "RENDER": _asset("RENDER", "Render (RNDR/RENDER)", 5690, "2020-06-11T10:10:00Z"),
    "SOL": _asset("SOL", "Solana", 5426, "2020-04-10T04:55:00Z"),
    "SUI": _asset("SUI", "Sui", 20947, "2023-05-03T12:05:00Z"),
    "TAO": _asset("TAO", "Bittensor", 22974, "2023-03-05T23:55:00Z"),
    "TRB": _asset("TRB", "Tellor", 4944, "2019-11-19T01:45:00Z"),
    "UNI": _asset("UNI", "Uniswap", 7083, "2020-09-17T01:10:00Z"),
    "WLD": _asset("WLD", "Worldcoin", 13502, "2023-07-24T07:25:00Z"),
    "XLM": _asset("XLM", "Stellar", 512, "2014-08-05T05:40:00Z"),
    "XMR": _asset("XMR", "Monero", 328, "2014-05-21T01:40:00Z"),
    "XRP": _asset("XRP", "XRP", 52, "2013-08-04T18:50:00Z"),
    "ZEC": _asset("ZEC", "Zcash", 1437, "2016-10-29T15:15:00Z"),
}


# Exact venue-native perpetual IDs, reviewed against the venue's listed
# instruments and underlying identities. Extend these rows only after reviewing
# the specific perpetual (or the venue's asset and its indexed contract).
# The same rendered ticker may represent different instruments across venues.
# OKX uses QUANT for the token and QNT for a TradFi perpetual:
# https://www.okx.com/en-br/help/okx-to-list-perpetual-futures-for-qnt-crypto
# https://www.okx.com/en-gb/help/okx-to-list-perpetual-futures-for-qnt-equity
# Other bindings verified by exact individual announcements:
# https://www.okx.com/en-br/help/okx-to-list-perpetual-futures-for-btw-crypto
# https://www.okx.com/en-eu/help/okx-to-list-perpetual-futures-for-aster-crypto
# https://www.okx.com/en-eu/help/okx-to-list-perpetual-futures-for-tao-crypto
# https://announcements.bybit.com/en/article/new-listing-btwusdt-perpetual-contract-with-up-to-10x-leverage-blt1cba31527e1ce290/
_MARKET_IDS = {
    "AAVE": {
        "binance": "AAVEUSDT",
        "bitget": "AAVEUSDT",
        "bitunix": "AAVEUSDT",
        "bybit": "AAVEUSDT",
        "gateio": "AAVE_USDT",
        "hyperliquid": "AAVE",
        "kucoin": "AAVEUSDTM",
        "okx": "AAVE-USDT-SWAP",
        "weex": "AAVEUSDT",
    },
    "ADA": {
        "binance": "ADAUSDT",
        "bitget": "ADAUSDT",
        "bitunix": "ADAUSDT",
        "bybit": "ADAUSDT",
        "gateio": "ADA_USDT",
        "kucoin": "ADAUSDTM",
        "okx": "ADA-USDT-SWAP",
        "weex": "ADAUSDT",
    },
    "ALGO": {
        "binance": "ALGOUSDT",
        "bitget": "ALGOUSDT",
        "bitunix": "ALGOUSDT",
        "bybit": "ALGOUSDT",
        "gateio": "ALGO_USDT",
        "hyperliquid": "ALGO",
        "kucoin": "ALGOUSDTM",
        "okx": "ALGO-USDT-SWAP",
        "weex": "ALGOUSDT",
    },
    "APT": {
        "binance": "APTUSDT",
        "bitget": "APTUSDT",
        "bitunix": "APTUSDT",
        "bybit": "APTUSDT",
        "gateio": "APT_USDT",
        "hyperliquid": "APT",
        "kucoin": "APTUSDTM",
        "okx": "APT-USDT-SWAP",
        "weex": "APTUSDT",
    },
    "ARB": {
        "binance": "ARBUSDT",
        "bitget": "ARBUSDT",
        "bitunix": "ARBUSDT",
        "bybit": "ARBUSDT",
        "gateio": "ARB_USDT",
        "hyperliquid": "ARB",
        "kucoin": "ARBUSDTM",
        "okx": "ARB-USDT-SWAP",
        "weex": "ARBUSDT",
    },
    "ASTER": {
        "binance": "ASTERUSDT",
        "bitget": "ASTERUSDT",
        "bitunix": "ASTERUSDT",
        "bybit": "ASTERUSDT",
        "gateio": "ASTER_USDT",
        "hyperliquid": "ASTER",
        "kucoin": "ASTERUSDTM",
        "okx": "ASTER-USDT-SWAP",
        "weex": "ASTERUSDT",
    },
    "ATOM": {
        "binance": "ATOMUSDT",
        "bitget": "ATOMUSDT",
        "bitunix": "ATOMUSDT",
        "bybit": "ATOMUSDT",
        "gateio": "ATOM_USDT",
        "hyperliquid": "ATOM",
        "kucoin": "ATOMUSDTM",
        "okx": "ATOM-USDT-SWAP",
        "weex": "ATOMUSDT",
    },
    "AVAX": {
        "binance": "AVAXUSDT",
        "bitget": "AVAXUSDT",
        "bitunix": "AVAXUSDT",
        "bybit": "AVAXUSDT",
        "gateio": "AVAX_USDT",
        "hyperliquid": "AVAX",
        "kucoin": "AVAXUSDTM",
        "okx": "AVAX-USDT-SWAP",
        "weex": "AVAXUSDT",
    },
    "BCH": {
        "binance": "BCHUSDT",
        "bitget": "BCHUSDT",
        "bitunix": "BCHUSDT",
        "bybit": "BCHUSDT",
        "gateio": "BCH_USDT",
        "hyperliquid": "BCH",
        "kucoin": "BCHUSDTM",
        "okx": "BCH-USDT-SWAP",
        "weex": "BCHUSDT",
    },
    "BNB": {
        "binance": "BNBUSDT",
        "bitget": "BNBUSDT",
        "bitunix": "BNBUSDT",
        "bybit": "BNBUSDT",
        "gateio": "BNB_USDT",
        "hyperliquid": "BNB",
        "kucoin": "BNBUSDTM",
        "okx": "BNB-USDT-SWAP",
        "weex": "BNBUSDT",
    },
    "BTC": {
        "binance": "BTCUSDT",
        "bitget": "BTCUSDT",
        "bitunix": "BTCUSDT",
        "bybit": "BTCUSDT",
        "gateio": "BTC_USDT",
        "hyperliquid": "BTC",
        "kucoin": "XBTUSDTM",
        "okx": "BTC-USDT-SWAP",
        "weex": "BTCUSDT",
    },
    "BTW": {
        "binance": "BTWUSDT",
        "bitget": "BTWUSDT",
        "bitunix": "BTWUSDT",
        "bybit": "BTWUSDT",
        "okx": "BTW-USDT-SWAP",
        "weex": "BTWUSDT",
    },
    "CC": {
        "binance": "CCUSDT",
        "bitget": "CCUSDT",
        "bitunix": "CCUSDT",
        "bybit": "CCUSDT",
        "gateio": "CC_USDT",
        "hyperliquid": "CC",
        "kucoin": "CCUSDTM",
        "okx": "CC-USDT-SWAP",
        "weex": "CCUSDT",
    },
    "CHZ": {
        "binance": "CHZUSDT",
        "bitget": "CHZUSDT",
        "bitunix": "CHZUSDT",
        "bybit": "CHZUSDT",
        "gateio": "CHZ_USDT",
        "kucoin": "CHZUSDTM",
        "okx": "CHZ-USDT-SWAP",
        "weex": "CHZUSDT",
    },
    "DOGE": {
        "binance": "DOGEUSDT",
        "bitget": "DOGEUSDT",
        "bitunix": "DOGEUSDT",
        "bybit": "DOGEUSDT",
        "gateio": "DOGE_USDT",
        "hyperliquid": "DOGE",
        "kucoin": "DOGEUSDTM",
        "okx": "DOGE-USDT-SWAP",
        "weex": "DOGEUSDT",
    },
    "DOT": {
        "binance": "DOTUSDT",
        "bitget": "DOTUSDT",
        "bitunix": "DOTUSDT",
        "bybit": "DOTUSDT",
        "gateio": "DOT_USDT",
        "hyperliquid": "DOT",
        "kucoin": "DOTUSDTM",
        "okx": "DOT-USDT-SWAP",
        "weex": "DOTUSDT",
    },
    "ENA": {
        "binance": "ENAUSDT",
        "bitget": "ENAUSDT",
        "bitunix": "ENAUSDT",
        "bybit": "ENAUSDT",
        "gateio": "ENA_USDT",
        "hyperliquid": "ENA",
        "kucoin": "ENAUSDTM",
        "okx": "ENA-USDT-SWAP",
        "weex": "ENAUSDT",
    },
    "ETC": {
        "binance": "ETCUSDT",
        "bitget": "ETCUSDT",
        "bitunix": "ETCUSDT",
        "bybit": "ETCUSDT",
        "gateio": "ETC_USDT",
        "hyperliquid": "ETC",
        "kucoin": "ETCUSDTM",
        "okx": "ETC-USDT-SWAP",
        "weex": "ETCUSDT",
    },
    "ETH": {
        "binance": "ETHUSDT",
        "bitget": "ETHUSDT",
        "bitunix": "ETHUSDT",
        "bybit": "ETHUSDT",
        "gateio": "ETH_USDT",
        "hyperliquid": "ETH",
        "kucoin": "ETHUSDTM",
        "okx": "ETH-USDT-SWAP",
        "weex": "ETHUSDT",
    },
    "FIL": {
        "binance": "FILUSDT",
        "bitget": "FILUSDT",
        "bitunix": "FILUSDT",
        "bybit": "FILUSDT",
        "gateio": "FIL_USDT",
        "hyperliquid": "FIL",
        "kucoin": "FILUSDTM",
        "okx": "FIL-USDT-SWAP",
        "weex": "FILUSDT",
    },
    "GRAM": {
        "binance": "GRAMUSDT",
        "bitget": "GRAMUSDT",
        "bitunix": "GRAMUSDT",
        "bybit": "GRAMUSDT",
        "hyperliquid": "GRAM",
        "kucoin": "GRAMUSDTM",
        "okx": "GRAM-USDT-SWAP",
        "weex": "GRAMUSDT",
    },
    "HBAR": {
        "binance": "HBARUSDT",
        "bitget": "HBARUSDT",
        "bitunix": "HBARUSDT",
        "bybit": "HBARUSDT",
        "gateio": "HBAR_USDT",
        "hyperliquid": "HBAR",
        "kucoin": "HBARUSDTM",
        "okx": "HBAR-USDT-SWAP",
        "weex": "HBARUSDT",
    },
    "HYPE": {
        "binance": "HYPEUSDT",
        "bitget": "HYPEUSDT",
        "bitunix": "HYPEUSDT",
        "bybit": "HYPEUSDT",
        "gateio": "HYPE_USDT",
        "hyperliquid": "HYPE",
        "kucoin": "HYPEUSDTM",
        "okx": "HYPE-USDT-SWAP",
        "weex": "HYPEUSDT",
    },
    "ICP": {
        "binance": "ICPUSDT",
        "bitget": "ICPUSDT",
        "bitunix": "ICPUSDT",
        "bybit": "ICPUSDT",
        "gateio": "ICP_USDT",
        "hyperliquid": "ICP",
        "kucoin": "ICPUSDTM",
        "okx": "ICP-USDT-SWAP",
        "weex": "ICPUSDT",
    },
    "INJ": {
        "binance": "INJUSDT",
        "bitget": "INJUSDT",
        "bitunix": "INJUSDT",
        "bybit": "INJUSDT",
        "gateio": "INJ_USDT",
        "hyperliquid": "INJ",
        "kucoin": "INJUSDTM",
        "okx": "INJ-USDT-SWAP",
        "weex": "INJUSDT",
    },
    "KAS": {
        "binance": "KASUSDT",
        "bitget": "KASUSDT",
        "bitunix": "KASUSDT",
        "bybit": "KASUSDT",
        "gateio": "KAS_USDT",
        "hyperliquid": "KAS",
        "kucoin": "KASUSDTM",
        "weex": "KASUSDT",
    },
    "LINK": {
        "binance": "LINKUSDT",
        "bitget": "LINKUSDT",
        "bitunix": "LINKUSDT",
        "bybit": "LINKUSDT",
        "gateio": "LINK_USDT",
        "hyperliquid": "LINK",
        "kucoin": "LINKUSDTM",
        "okx": "LINK-USDT-SWAP",
        "weex": "LINKUSDT",
    },
    "LTC": {
        "binance": "LTCUSDT",
        "bitget": "LTCUSDT",
        "bitunix": "LTCUSDT",
        "bybit": "LTCUSDT",
        "gateio": "LTC_USDT",
        "hyperliquid": "LTC",
        "kucoin": "LTCUSDTM",
        "okx": "LTC-USDT-SWAP",
        "weex": "LTCUSDT",
    },
    "MORPHO": {
        "binance": "MORPHOUSDT",
        "bitget": "MORPHOUSDT",
        "bitunix": "MORPHOUSDT",
        "bybit": "MORPHOUSDT",
        "gateio": "MORPHO_USDT",
        "hyperliquid": "MORPHO",
        "kucoin": "MORPHOUSDTM",
        "okx": "MORPHO-USDT-SWAP",
        "weex": "MORPHOUSDT",
    },
    "NEAR": {
        "binance": "NEARUSDT",
        "bitget": "NEARUSDT",
        "bitunix": "NEARUSDT",
        "bybit": "NEARUSDT",
        "gateio": "NEAR_USDT",
        "hyperliquid": "NEAR",
        "kucoin": "NEARUSDTM",
        "okx": "NEAR-USDT-SWAP",
        "weex": "NEARUSDT",
    },
    "ONDO": {
        "binance": "ONDOUSDT",
        "bitget": "ONDOUSDT",
        "bitunix": "ONDOUSDT",
        "bybit": "ONDOUSDT",
        "gateio": "ONDO_USDT",
        "hyperliquid": "ONDO",
        "kucoin": "ONDOUSDTM",
        "okx": "ONDO-USDT-SWAP",
        "weex": "ONDOUSDT",
    },
    "OP": {
        "binance": "OPUSDT",
        "bitget": "OPUSDT",
        "bitunix": "OPUSDT",
        "bybit": "OPUSDT",
        "gateio": "OP_USDT",
        "hyperliquid": "OP",
        "kucoin": "OPUSDTM",
        "okx": "OP-USDT-SWAP",
        "weex": "OPUSDT",
    },
    "POL": {
        "binance": "POLUSDT",
        "bitget": "POLUSDT",
        "bitunix": "POLUSDT",
        "bybit": "POLUSDT",
        "gateio": "POL_USDT",
        "hyperliquid": "POL",
        "kucoin": "POLUSDTM",
        "okx": "POL-USDT-SWAP",
        "weex": "POLUSDT",
    },
    "QNT": {
        "binance": "QNTUSDT",
        "bitget": "QNTUSDT",
        "bitunix": "QNTUSDT",
        "bybit": "QNTUSDT",
        "gateio": "QNT_USDT",
        "kucoin": "QNTUSDTM",
        "okx": "QUANT-USDT-SWAP",
        "weex": "QNTUSDT",
    },
    "RENDER": {
        "binance": "RENDERUSDT",
        "bitget": "RENDERUSDT",
        "bitunix": "RENDERUSDT",
        "bybit": "RENDERUSDT",
        "hyperliquid": "RENDER",
        "kucoin": "RENDERUSDTM",
        "okx": "RENDER-USDT-SWAP",
        "weex": "RENDERUSDT",
    },
    "SOL": {
        "binance": "SOLUSDT",
        "bitget": "SOLUSDT",
        "bitunix": "SOLUSDT",
        "bybit": "SOLUSDT",
        "gateio": "SOL_USDT",
        "hyperliquid": "SOL",
        "kucoin": "SOLUSDTM",
        "okx": "SOL-USDT-SWAP",
        "weex": "SOLUSDT",
    },
    "SUI": {
        "binance": "SUIUSDT",
        "bitget": "SUIUSDT",
        "bitunix": "SUIUSDT",
        "bybit": "SUIUSDT",
        "gateio": "SUI_USDT",
        "hyperliquid": "SUI",
        "kucoin": "SUIUSDTM",
        "okx": "SUI-USDT-SWAP",
        "weex": "SUIUSDT",
    },
    "TAO": {
        "binance": "TAOUSDT",
        "bitget": "TAOUSDT",
        "bitunix": "TAOUSDT",
        "bybit": "TAOUSDT",
        "gateio": "TAO_USDT",
        "hyperliquid": "TAO",
        "kucoin": "TAOUSDTM",
        "okx": "TAO-USDT-SWAP",
        "weex": "TAOUSDT",
    },
    "TRB": {
        "binance": "TRBUSDT",
        "bitget": "TRBUSDT",
        "bitunix": "TRBUSDT",
        "bybit": "TRBUSDT",
        "gateio": "TRB_USDT",
        "hyperliquid": "TRB",
        "kucoin": "TRBUSDTM",
        "okx": "TRB-USDT-SWAP",
        "weex": "TRBUSDT",
    },
    "UNI": {
        "binance": "UNIUSDT",
        "bitget": "UNIUSDT",
        "bitunix": "UNIUSDT",
        "bybit": "UNIUSDT",
        "gateio": "UNI_USDT",
        "hyperliquid": "UNI",
        "kucoin": "UNIUSDTM",
        "okx": "UNI-USDT-SWAP",
        "weex": "UNIUSDT",
    },
    "WLD": {
        "binance": "WLDUSDT",
        "bitget": "WLDUSDT",
        "bitunix": "WLDUSDT",
        "bybit": "WLDUSDT",
        "gateio": "WLD_USDT",
        "hyperliquid": "WLD",
        "kucoin": "WLDUSDTM",
        "okx": "WLD-USDT-SWAP",
        "weex": "WLDUSDT",
    },
    "XLM": {
        "binance": "XLMUSDT",
        "bitget": "XLMUSDT",
        "bitunix": "XLMUSDT",
        "bybit": "XLMUSDT",
        "gateio": "XLM_USDT",
        "hyperliquid": "XLM",
        "kucoin": "XLMUSDTM",
        "okx": "XLM-USDT-SWAP",
        "weex": "XLMUSDT",
    },
    "XMR": {
        "binance": "XMRUSDT",
        "bitget": "XMRUSDT",
        "bybit": "XMRUSDT",
        "hyperliquid": "XMR",
        "kucoin": "XMRUSDTM",
        "weex": "XMRUSDT",
    },
    "XRP": {
        "binance": "XRPUSDT",
        "bitget": "XRPUSDT",
        "bitunix": "XRPUSDT",
        "bybit": "XRPUSDT",
        "gateio": "XRP_USDT",
        "hyperliquid": "XRP",
        "kucoin": "XRPUSDTM",
        "okx": "XRP-USDT-SWAP",
        "weex": "XRPUSDT",
    },
    "ZEC": {
        "binance": "ZECUSDT",
        "bitget": "ZECUSDT",
        "bybit": "ZECUSDT",
        "gateio": "ZEC_USDT",
        "hyperliquid": "ZEC",
        "kucoin": "ZECUSDTM",
        "okx": "ZEC-USDT-SWAP",
        "weex": "ZECUSDT",
    },
}


# Market base codes that differ from the canonical asset ticker.
# https://www.binance.com/en/support/announcement/detail/fe307fba935b44698fde4db01e84a7eb
# https://www.okx.com/en-gb/help/okx-to-support-matic-token-migration
# https://www.okx.com/en-us/help/okx-to-enable-margin-trading-and-simple-earn-and-list-perpetual-for-rndr
# https://www.binance.com/en/support/announcement/detail/499af13827e04741999c15f9139ff3de
# https://www.binance.com/en/support/announcement/detail/17595222999545348c8acdd5dc41c00a
_MARKET_ASSETS = {
    (exchange, native_id): (asset, "QUANT" if native_id == "QUANT-USDT-SWAP" else asset)
    for asset, rows in _MARKET_IDS.items()
    for exchange, native_id in rows.items()
}
_MARKET_ASSETS.update(
    {
        ("binance", "TONUSDT"): ("GRAM", "TON"),
        ("okx", "MATIC-USDT-SWAP"): ("POL", "MATIC"),
        ("okx", "RNDR-USDT-SWAP"): ("RENDER", "RNDR"),
        ("binance", "RNDRUSDT"): ("RENDER", "RNDR"),
        ("binance", "LENDUSDT"): ("AAVE", "LEND"),
    }
)

# Restrict current OKX bindings to the observed instrument generation when
# the public market snapshot includes instIdCode. Older snapshots and delisted
# migration instruments may carry only the reviewed native ID.
_OKX_INSTRUMENT_CODES = {
    "AAVE-USDT-SWAP": 10485,
    "ADA-USDT-SWAP": 10483,
    "ALGO-USDT-SWAP": 10487,
    "APT-USDT-SWAP": 104900,
    "ARB-USDT-SWAP": 117782,
    "ASTER-USDT-SWAP": 228297,
    "ATOM-USDT-SWAP": 10490,
    "AVAX-USDT-SWAP": 10491,
    "BCH-USDT-SWAP": 10465,
    "BNB-USDT-SWAP": 111061,
    "BTC-USDT-SWAP": 10459,
    "BTW-USDT-SWAP": 523548,
    "CC-USDT-SWAP": 234190,
    "CHZ-USDT-SWAP": 16651,
    "DOGE-USDT-SWAP": 10502,
    "DOT-USDT-SWAP": 10504,
    "ENA-USDT-SWAP": 227622,
    "ETC-USDT-SWAP": 10469,
    "ETH-USDT-SWAP": 10461,
    "FIL-USDT-SWAP": 10507,
    "GRAM-USDT-SWAP": 318896,
    "HBAR-USDT-SWAP": 129258,
    "HYPE-USDT-SWAP": 198413,
    "ICP-USDT-SWAP": 21505,
    "INJ-USDT-SWAP": 139329,
    "LINK-USDT-SWAP": 10519,
    "LTC-USDT-SWAP": 10463,
    "MORPHO-USDT-SWAP": 186041,
    "NEAR-USDT-SWAP": 10523,
    "ONDO-USDT-SWAP": 170039,
    "OP-USDT-SWAP": 86647,
    "POL-USDT-SWAP": 175501,
    "QUANT-USDT-SWAP": 488922,
    "RENDER-USDT-SWAP": 170252,
    "SOL-USDT-SWAP": 10530,
    "SUI-USDT-SWAP": 120850,
    "TAO-USDT-SWAP": 174952,
    "TRB-USDT-SWAP": 10540,
    "UNI-USDT-SWAP": 10543,
    "WLD-USDT-SWAP": 127411,
    "XLM-USDT-SWAP": 10481,
    "XRP-USDT-SWAP": 10471,
    "ZEC-USDT-SWAP": 234991,
}


_EXCHANGE_ALIASES = {
    "binanceusdm": "binance",
    "gate": "gateio",
    "kucoinfutures": "kucoin",
}


def _market_symbol(exchange: str, base: str) -> str:
    currency = "USDC" if exchange == "hyperliquid" else "USDT"
    return f"{base}/{currency}:{currency}"


_BINDINGS_BY_ASSET = defaultdict(list)
for (venue, native_market_id), (asset_code, base_code) in _MARKET_ASSETS.items():
    _BINDINGS_BY_ASSET[asset_code].append(
        (venue, native_market_id, _market_symbol(venue, base_code))
    )


def asset_venue_bindings(identity: AssetIdentity) -> tuple[tuple[str, str, str], ...]:
    """Return the reviewed (venue, native ID, symbol) matches for an asset."""
    if _ASSETS.get(identity.cmc_symbol) != identity:
        return ()
    return tuple(_BINDINGS_BY_ASSET[identity.cmc_symbol])


def is_non_crypto_instrument(exchange: str, market: Mapping) -> bool:
    """Reject known non-crypto underlyings before resolving asset age."""
    info = market.get("info")
    info = info if isinstance(info, Mapping) else {}
    if exchange == "okx":
        # Some previous venue snapshots have category but no instCategory.
        # The former can be "1" for both crypto and stock products.
        return (
            market.get("id") == "QNT-USDT-SWAP"
            or market.get("symbol") == "QNT/USDT:USDT"
            or info.get("instCategory") not in (None, "", "1", 1)
        )
    if exchange == "weex":
        return info.get("contractType") == "TRADIFI_PERPETUAL" or info.get(
            "underlyingType"
        ) not in (None, "", "COIN")
    if exchange == "kucoin":
        return any(
            info.get(field) not in (None, "", "CRYPTO")
            for field in ("assetClass", "marketType")
        )
    return info.get("isRwa") in (True, "YES") or info.get("marketType") in (
        "STOCK",
        "METAL",
        "COMMODITY",
    )


def resolve_asset_identity(exchange: str, market: Mapping) -> AssetIdentity | None:
    """Return an identity only for a reviewed exact venue perpetual."""
    exchange = _EXCHANGE_ALIASES.get(exchange, exchange)
    if is_non_crypto_instrument(exchange, market):
        return None
    native_id = market.get("id")
    if not isinstance(native_id, str):
        return None
    match = _MARKET_ASSETS.get((exchange, native_id))
    if match is None:
        return None
    asset, base = match
    currency = "USDC" if exchange == "hyperliquid" else "USDT"
    if (
        market.get("swap") is not True
        or market.get("base") != base
        or market.get("symbol") != _market_symbol(exchange, base)
        or market.get("quote") != currency
        or market.get("settle") != currency
    ):
        return None
    info = market.get("info")
    if isinstance(info, Mapping):
        code = info.get("instIdCode")
        expected_code = _OKX_INSTRUMENT_CODES.get(native_id) if exchange == "okx" else None
        if expected_code is not None and code not in (None, "", expected_code, str(expected_code)):
            return None
        if info.get("isPreListing") is True or info.get("is_pre_market") is True:
            return None
    return _ASSETS[asset]
