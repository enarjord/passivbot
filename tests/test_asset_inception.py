from datetime import datetime, timezone
import json

import pytest

import asset_inception as age
from asset_inception_catalog import AssetIdentity
from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore
from simulation_data import data_manifest, simulation_data_policy


def ms(text):
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp() * 1_000)


START = ms("2020-01-01")
NOW = ms("2020-02-20")
DAY = age.DAY_MS


def okx_market(base="TEST", **info):
    return {
        "id": f"{base}-USDT-SWAP",
        "symbol": f"{base}/USDT:USDT",
        "base": base,
        "quote": "USDT",
        "settle": "USDT",
        "swap": True,
        "info": {
            "instId": f"{base}-USDT-SWAP",
            "state": "live",
            "ruleType": "normal",
            "instCategory": "1",
            "listTime": str(START),
            "preMktSwTime": "",
            **info,
        },
    }


class JsonResponse:
    def __init__(self, data):
        self.data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    def raise_for_status(self):
        return None

    async def json(self):
        return self.data


class CMCSession:
    def __init__(self, rows):
        self.rows = rows

    def get(self, path, *, params, allow_redirects):
        assert path.endswith("/public-api/v1/cryptocurrency/map")
        assert params["symbol"] == "TEST"
        assert not allow_redirects
        return JsonResponse({"status": {"error_code": 0}, "data": self.rows})


@pytest.mark.parametrize(
    "rows, expected",
    [
        (
            [
                {"id": 11, "symbol": "TEST", "first_historical_data": "2020-01-02T00:00:00Z"},
                {"id": 12, "symbol": "TEST", "first_historical_data": "2019-01-01T00:00:00Z"},
            ],
            {11: START + DAY},
        ),
        ([{"id": 11, "symbol": "TEST", "first_historical_data": "2019-01-02T00:00:00Z"}], {}),
        ([{"id": 11, "symbol": "TEST", "first_historical_data": "2020-01-02T00:00:00"}], {}),
        ([{"id": 11, "symbol": "TEST", "first_historical_data": "2021-01-02T00:00:00Z"}], {}),
        ([{"id": 11, "symbol": "ALIASED", "first_historical_data": "2020-01-02T00:00:00Z"}], {}),
        (
            [
                {"id": 11, "symbol": "TEST", "first_historical_data": "2020-01-02T00:00:00Z"},
                {"id": 11, "symbol": "TEST", "first_historical_data": "2020-01-03T00:00:00Z"},
            ],
            {},
        ),
    ],
)
async def test_coinmarketcap_requires_exact_identity_and_postlaunch_price(rows, expected):
    identity = AssetIdentity("Test", 11, "TEST", START)
    assert await age._fetch_cmc(CMCSession(rows), [identity], NOW) == expected


async def test_offline_admission_never_calls_provider_or_venue(tmp_path, monkeypatch):
    ident = AssetIdentity("Test", 11, "TEST", START)
    market = okx_market()

    def fail(*args, **kwargs):
        pytest.fail("unexpected public market data request")

    monkeypatch.setattr(age, "_resolve_cmc", fail)
    monkeypatch.setattr(age, "_resolve_venue", fail)
    monkeypatch.setattr(age, "resolve_asset_identity", lambda *_: ident)
    with simulation_data_policy({"backtest": {"offline": True}}):
        assert await age.resolve_asset_inception({"TEST": [("okx", market)]}, cache_dir=tmp_path) == {}


async def test_prelaunch_and_stock_markets_cannot_acquire_age(tmp_path, monkeypatch):
    converted = okx_market("TEST", preMktSwTime=str(START + DAY))
    assert age._regular_trading_floor("okx", converted, None, NOW) == START + DAY
    assert age._regular_trading_floor("okx", okx_market(ruleType="xperp"), None, NOW) is None

    def fail(*args, **kwargs):
        pytest.fail("unexpected public market data request")

    monkeypatch.setattr(age, "_resolve_venue", fail)
    assert await age.resolve_asset_inception(
        {
            # The older metadata shape omitted instCategory for stocks.
            "QNT": [("okx", okx_market("QNT", instCategory=None))],
            "TEST": [("okx", okx_market(ruleType="pre_market"))],
        },
        cache_dir=tmp_path,
    ) == {}


@pytest.mark.parametrize("exchange, raw", [
    ("kucoin", {
        "symbol": "TESTUSDTM",
        "status": "Open",
        "marketStage": "NORMAL",
        "marketType": "CRYPTO",
        "firstOpenDate": START,
        "preMarketToPerpDate": START + DAY,
    }),
    ("bitunix", {"symbol": "TESTUSDTM", "symbolStatus": "OPEN", "launchTime": START + DAY}),
])
def test_normal_trading_floor_matches_exact_venue_contract(exchange, raw):
    market = {"id": "TESTUSDTM", "swap": True, "info": raw}
    identity = AssetIdentity("Test", 11, "TEST", START)
    assert age._regular_trading_floor(exchange, market, identity, NOW) == START + DAY
    assert age._regular_trading_floor(exchange, {**market, "id": "FAKEUSDT"}, identity, NOW) is None


def test_unmapped_market_requires_crypto_classification():
    market = okx_market(instCategory=None)
    assert age._regular_trading_floor("okx", market, None, NOW) is None
    assert age._regular_trading_floor(
        "bitunix",
        {"id": "TESTUSDT", "swap": True, "info": {
            "symbol": "TESTUSDT", "symbolStatus": "OPEN", "launchTime": START,
        }},
        None, NOW,
    ) is None


async def test_venue_fallback_uses_completed_positive_volume_after_conversion(tmp_path, monkeypatch):
    market = okx_market(preMktSwTime=str(START + DAY + 1))

    class Venue:
        async def fetch_ohlcv(self, symbol, *, since, timeframe, limit, params):
            assert symbol == market["symbol"]
            if timeframe == "1m":
                assert since == START + DAY + age.MINUTE_MS
                return []
            assert timeframe == "1d"
            assert since == START + 2 * DAY
            assert params["type"] == "HistoryCandles"
            return [
                [START + DAY, 2, 4, 1, 3, 4],  # straddles normal trading
                [START + 2 * DAY, 2, 4, 1, 3, 0],
                [START + 3 * DAY, 2, 4, 1, 3, 5],
            ]

        async def close(self):
            self.closed = True

    client = Venue()
    monkeypatch.setattr(age, "_new_venue_client", lambda *_: client)
    request = {"TEST": [("okx", market)]}
    evidence = await age.resolve_asset_inception(request, cache_dir=tmp_path)
    assert evidence == {"TEST": age.InceptionEvidence(START + 4 * DAY, "venue_candle:okx", None, True)}
    assert client.closed
    assert evidence == await age.resolve_asset_inception(request, allow_remote=False, cache_dir=tmp_path)


async def test_venue_recheck_preserves_then_improves_provisional_age(tmp_path, monkeypatch):
    request = {"TEST": [("okx", okx_market())]}
    earliest = START + age.MINUTE_MS
    observations = iter([START + DAY, age.ccxt.NetworkError("temporary"), earliest, START + DAY])

    async def fetch(*args):
        observation = next(observations)
        if isinstance(observation, Exception):
            raise observation
        return observation

    monkeypatch.setattr(age, "_fetch_venue_candle", fetch)
    expected = [START + DAY, START + DAY, earliest, earliest]
    for timestamp in expected:
        result = await age.resolve_asset_inception(request, cache_dir=tmp_path)
        assert result["TEST"] == age.InceptionEvidence(timestamp, "venue_candle:okx", None, True)
    offline = await age.resolve_asset_inception(request, allow_remote=False, cache_dir=tmp_path)
    assert offline["TEST"].timestamp_ms == earliest


async def test_later_provider_observation_keeps_earlier_provisional_age_refreshable(tmp_path, monkeypatch):
    identity = AssetIdentity("Test", 11, "TEST", START)
    monkeypatch.setattr(age, "resolve_asset_identity", lambda *_: identity)

    async def provider(*args):
        for fact in provider_facts:
            yield fact

    provider_facts = []
    monkeypatch.setattr(age, "_resolve_cmc", provider)
    observations = iter([START + DAY, START + age.MINUTE_MS])

    async def fetch(*args):
        return next(observations)

    monkeypatch.setattr(age, "_fetch_venue_candle", fetch)
    request = {"TEST": [("okx", okx_market())]}
    old = await age.resolve_asset_inception(request, cache_dir=tmp_path)
    assert old["TEST"].timestamp_ms == START + DAY
    provider_facts.append(age.InceptionEvidence(START + 2 * DAY, "coinmarketcap:11", "coinmarketcap:11"))
    improved = await age.resolve_asset_inception(request, cache_dir=tmp_path)
    assert improved["TEST"] == age.InceptionEvidence(
        START + age.MINUTE_MS, "venue_candle:okx", "coinmarketcap:11", True
    )


async def test_cross_venue_local_earlier_evidence_is_verified_and_fingerprinted(tmp_path, monkeypatch):
    ident = AssetIdentity("Test", 11, "TEST", START, START + 2 * DAY, "https://example.com/price")
    market = okx_market(preMktSwTime=str(START + age.MINUTE_MS))
    binance = {"id": "TESTUSDT", **{k: market[k] for k in ("base", "symbol", "swap")}}

    monkeypatch.setattr(age, "resolve_asset_identity", lambda *_: ident)
    monkeypatch.setattr(age, "asset_venue_bindings", lambda _: [("okx", market["id"], market["symbol"])])
    (tmp_path / "okx").mkdir()
    (tmp_path / "okx" / "markets.json").write_text(json.dumps({market["symbol"]: market}))
    root = tmp_path / "ohlcvs"
    store = OhlcvStore(root, OhlcvCatalog(root / "catalog.sqlite"))
    store.write_rows(
        "okx", "1m", market["symbol"], [START, START + age.MINUTE_MS, START + 2 * age.MINUTE_MS],
        [[4, 1, 3, 10], [4, 1, 3, 0], [4, 1, 3, 2]],
    )
    request = {"TEST": [("binance", binance)]}
    with simulation_data_policy({"backtest": {"offline": True}}):
        facts = await age.resolve_asset_inception(request, cache_dir=tmp_path)
        state = data_manifest()
    assert facts == {
        "TEST": age.InceptionEvidence(START + 3 * age.MINUTE_MS, "local_candle:okx", "coinmarketcap:11", True)
    }
    assert str(tmp_path / "okx" / "markets.json") in state["metadata"]
    assert state["ranges"][0]["actual_start_ms"] == START + 2 * age.MINUTE_MS


async def test_combined_matches_cannot_share_an_unverified_symbol(tmp_path, monkeypatch):
    identity = AssetIdentity("Test", 11, "TEST", START, START + DAY, "https://example.com/price")
    monkeypatch.setattr(age, "resolve_asset_identity", lambda venue, _: identity if venue == "okx" else None)
    market = okx_market()
    assert await age.resolve_asset_inception(
        {"TEST": [("okx", market), ("bybit", market)]}, cache_dir=tmp_path
    ) == {}


async def test_provider_fact_survives_offline_and_beats_venue_fallback(tmp_path, monkeypatch):
    ident = AssetIdentity("Test", 11, "TEST", START)
    monkeypatch.setattr(age, "resolve_asset_identity", lambda *_: ident)

    class Session(CMCSession):
        def __init__(self, **_):
            super().__init__(
                [{"id": 11, "symbol": "TEST", "first_historical_data": "2020-01-02T00:00:00Z"}]
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

    monkeypatch.setattr(age.aiohttp, "ClientSession", Session)

    def fail(*args, **kwargs):
        pytest.fail("unexpected fall-through to venue data")

    monkeypatch.setattr(age, "_resolve_venue", fail)
    request = {"TEST": [("okx", okx_market())]}
    expected = {
        "TEST": age.InceptionEvidence(START + DAY, "coinmarketcap:11", "coinmarketcap:11")
    }
    assert await age.resolve_asset_inception(request, cache_dir=tmp_path) == expected
    monkeypatch.setattr(age, "_resolve_cmc", fail)
    with simulation_data_policy({"backtest": {"offline": True}}):
        assert await age.resolve_asset_inception(request, cache_dir=tmp_path) == expected
        assert len(data_manifest()["metadata"]) == 1


def test_local_chunk_snapshot_refuses_modified_body_without_creating_lock(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "catalog.sqlite")
    store = OhlcvStore(tmp_path, catalog)
    symbol = okx_market()["symbol"]
    store.write_rows("okx", "1m", symbol, [START], [[4, 1, 3, 2]])
    chunk, = catalog.list_chunks("okx", "1m", symbol, START, START)
    for lock in tmp_path.rglob("*.lock"):
        lock.unlink()

    assert age._local_exchange_candle(store, "okx", symbol, START, START + DAY) == START + age.MINUTE_MS
    assert not list(tmp_path.rglob("*.lock"))
    data = age.np.load(chunk.body_path, mmap_mode="r+")
    data[0, 3] = 90
    data.flush()
    del data
    with pytest.raises(ValueError, match="catalog snapshot"):
        age._local_exchange_candle(store, "okx", symbol, START, START + DAY)
    assert not list(tmp_path.rglob("*.lock"))


@pytest.mark.parametrize(
    ("venue", "native_id", "info"),
    [
        (
            # https://www.gate.com/es/announcements/article/47933
            "gateio", "CC_USDT",
            {"name": "CC_USDT", "status": "trading", "is_pre_market": False, "launch_time": 1761913256},
        ),
        (
            # https://www.bitget.com/support/articles/12560603846514
            "bitget", "LITUSDT",
            {"symbol": "LITUSDT", "symbolStatus": "normal", "openTime": "1766559843478", "launchTime": ""},
        ),
    ],
)
def test_converted_gate_and_bitget_markets_retain_prelaunch_times(venue, native_id, info):
    now = ms("2026-11-15")
    market = {"id": native_id, "swap": True, "info": info}
    assert age._regular_trading_floor(venue, market, None, now) is None
