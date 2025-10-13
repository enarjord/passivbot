import json

import pytest

from custom_endpoint_overrides import (
    CustomEndpointConfig,
    ResolvedEndpointOverride,
    apply_rest_overrides_to_ccxt,
    configure_custom_endpoint_loader,
    get_cached_custom_endpoint_config,
    load_custom_endpoint_config,
    resolve_custom_endpoint_override,
)


@pytest.fixture(autouse=True)
def reset_custom_endpoint_loader():
    """
    Ensure each test starts with a clean loader configuration.
    """
    configure_custom_endpoint_loader(None, autodiscover=True, preloaded=None)
    yield
    configure_custom_endpoint_loader(None, autodiscover=True, preloaded=None)


def _write_config(path, content):
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_load_custom_endpoint_config_parses_binance_override(tmp_path):
    config_path = _write_config(
        tmp_path / "custom_endpoints.json",
        {
            "defaults": {"rest": {"rewrite_domains": {}, "url_overrides": {}, "extra_headers": {}}},
            "exchanges": {
                "binanceusdm": {
                    "disable_ws": True,
                    "rest": {
                        "rewrite_domains": {
                            "https://fapi.binance.com": "https://proxy.example.exchange"
                        }
                    },
                }
            },
        },
    )

    config = load_custom_endpoint_config(str(config_path))
    override = config.get_override("binanceusdm")

    assert override is not None
    assert override.disable_ws is True
    assert override.rest_domain_rewrites == {
        "https://fapi.binance.com": "https://proxy.example.exchange"
    }
    assert override.rest_url_overrides == {}
    assert override.rest_extra_headers == {}
    assert config.get_override("unknown") is None


def test_apply_rest_overrides_to_ccxt_updates_urls_and_headers():
    override = ResolvedEndpointOverride(
        exchange_id="binanceusdm",
        rest_domain_rewrites={"https://fapi.binance.com": "https://proxy.example"},
        rest_url_overrides={"fapiPrivate": "https://proxy.example/fapi/v1"},
        rest_extra_headers={"X-Test": "1"},
        disable_ws=True,
    )

    class DummyExchange:
        def __init__(self):
            self.urls = {
                "api": {
                    "fapiPrivate": "https://fapi.binance.com/fapi/v1",
                    "fapiPublic": "https://fapi.binance.com/fapi/v1",
                    "public": "https://api.binance.com/api/v3",
                }
            }
            self.headers = {"existing": "header"}

    exchange = DummyExchange()
    apply_rest_overrides_to_ccxt(exchange, override)

    assert exchange.urls["api"]["fapiPrivate"] == "https://proxy.example/fapi/v1"
    assert exchange.urls["api"]["fapiPublic"] == "https://proxy.example/fapi/v1"
    assert exchange.urls["api"]["public"] == "https://api.binance.com/api/v3"
    assert exchange.headers["existing"] == "header"
    assert exchange.headers["X-Test"] == "1"


def test_apply_rest_overrides_handles_hostname_placeholder():
    override = ResolvedEndpointOverride(
        exchange_id="bybit",
        rest_domain_rewrites={"https://api.{hostname}": "https://bybit-proxy"},
        disable_ws=True,
    )

    class DummyBybit:
        def __init__(self):
            self.hostname = "bybit.com"
            self.urls = {
                "api": {"public": "https://api.{hostname}/v5", "private": "https://api.{hostname}/v5"}
            }
            self.headers = {}

    exchange = DummyBybit()
    apply_rest_overrides_to_ccxt(exchange, override)

    assert exchange.urls["api"]["public"] == "https://bybit-proxy/v5"
    assert exchange.urls["api"]["private"] == "https://bybit-proxy/v5"


def test_configure_loader_disables_autodiscovery():
    configure_custom_endpoint_loader(None, autodiscover=False, preloaded=None)

    config = get_cached_custom_endpoint_config()
    assert isinstance(config, CustomEndpointConfig)
    assert config.is_empty()
    assert resolve_custom_endpoint_override("binanceusdm") is None


def test_configure_loader_with_explicit_path(tmp_path):
    config_path = _write_config(
        tmp_path / "custom_endpoints.json",
        {
            "exchanges": {
                "binanceusdm": {
                    "disable_ws": False,
                    "rest": {
                        "rewrite_domains": {"https://fapi.binance.com": "https://proxy.example"},
                        "url_overrides": {"fapiPrivate": "https://proxy.example/fapi/v1"},
                    },
                }
            }
        },
    )

    configure_custom_endpoint_loader(str(config_path), autodiscover=False, preloaded=None)
    override = resolve_custom_endpoint_override("binanceusdm")

    assert override is not None
    assert override.disable_ws is False
    assert override.rest_url_overrides["fapiPrivate"] == "https://proxy.example/fapi/v1"
    assert override.rest_domain_rewrites["https://fapi.binance.com"] == "https://proxy.example"
