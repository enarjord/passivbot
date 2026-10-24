"""Inspection resolves inheritance without changing reusable source authority."""

from copy import deepcopy

import pytest

from config.authoring import compact_config, explain_config, resolved_config
from config.load import prepare_config
from config.schema import get_template_config


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_compact_defaults_do_not_introduce_side_activation_pins(kind):
    source = get_template_config()
    source["bot"]["portfolio"]["strategy_kind"] = kind
    before = deepcopy(source)
    compact = compact_config(source)
    assert compact["bot"]["long"] == compact["bot"]["short"] == {}
    compact["bot"].setdefault("long", {})["enabled"] = True
    compact["bot"].setdefault("short", {})["enabled"] = True
    loaded = prepare_config(compact, verbose=False)
    assert loaded["bot"]["long"]["enabled"] is True
    assert loaded["bot"]["short"]["enabled"] is True
    assert source == before


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_explain_resolves_shared_gene_and_equal_fixed_side_pin(kind):
    source = {"config_version": "v9.0.0", "bot": {
        "portfolio": {"strategy_kind": kind},
        "global": {"risk": {"total_wallet_exposure_limit": 1.5}},
        "short": {"risk": {"total_wallet_exposure_limit": 1.5}},
    }, "optimize": {"bounds": {
        "global": {"risk": {"total_wallet_exposure_limit": [0.5, 2.5, 0.1]}},
    }}}
    before = deepcopy(source)
    inherited = explain_config(source, "bot.long.risk.total_wallet_exposure_limit")
    pinned = explain_config(source, "bot.short.risk.total_wallet_exposure_limit")
    assert inherited["search"]["key"] == "bot.global.risk.total_wallet_exposure_limit"
    assert inherited["search"]["status"] == "searched"
    assert inherited["search"]["bounds"] == [0.5, 2.5, 0.1]
    assert inherited["sources"][-1]["kind"] == "inherited"
    assert pinned["search"]["key"] == "bot.short.risk.total_wallet_exposure_limit"
    assert pinned["search"]["status"] == "omitted_fixed"
    assert pinned["search"]["bounds"] == [1.5, 1.5]
    domain = explain_config(source, "optimize.bounds.global.risk.total_wallet_exposure_limit")
    assert domain["value"] == [0.5, 2.5, 0.1]
    assert domain["bound_target"] == "bot.global.risk.total_wallet_exposure_limit"
    for export in (compact_config, resolved_config):
        output = export(source)
        assert "risk" not in output["bot"].get("long", {})
        assert output["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 1.5
    assert source == before
