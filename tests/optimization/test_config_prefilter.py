from copy import deepcopy
from types import SimpleNamespace

import pytest

from config.prefilter import resolve_prefilter
from config.scoring import extract_objective_specs
from config_utils import clean_config, get_template_config
from optimization.prefilter import (
    build_prefilter_payload,
    config_prefilter,
    inspect_config_prefilter,
    prefilter_candidate,
    prefilter_config_for_exchange,
)


@pytest.fixture
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires the real Rust extension")
    verify_loaded_runtime_extension()
    return pbr


def config():
    result = get_template_config()
    result["backtest"]["coins"] = {"binance": ["TEST"]}
    result["live"]["approved_coins"] = {"long": ["TEST"], "short": []}
    side = result["bot"]["long"]
    side["risk"].update(
        n_positions=1, total_wallet_exposure_limit=1.0, we_excess_allowance_pct=0.0
    )
    side["entry_cooldown"].update(base_duration_minutes=5.0)
    strategy = side["strategy"]["trailing_martingale"]
    strategy["entry"].update(
        initial_qty_pct=0.1,
        double_down_factor=1.0,
        threshold_base_pct=0.03,
        threshold_we_weight=5.0,
        retracement_base_pct=0.001,
        retracement_we_weight=0.1,
        threshold_volatility_1m_weight=0,
        threshold_volatility_1h_weight=0,
        retracement_volatility_1m_weight=0,
        retracement_volatility_1h_weight=0,
    )
    strategy["close"].update(
        threshold_base_pct=0.01,
        threshold_we_weight=-0.005,
        threshold_volatility_1m_weight=0,
        threshold_volatility_1h_weight=0,
        retracement_base_pct=0.002,
        retracement_volatility_1m_weight=0,
        retracement_volatility_1h_weight=0,
    )
    return result


def policy(metric="entry_close_span", **bounds):
    return resolve_prefilter(
        {
            "exposure_ratios": [0.8],
            "checks": [
                {"metric": metric, "enabled": True, "sides": ["long"], **bounds},
            ],
        }
    )


def test_every_check_is_disabled_unless_explicitly_enabled(monkeypatch):
    import passivbot_rust as pbr

    def forbidden(*_args):
        pytest.fail("disabled policy must not call Rust or inspect strategy inputs")

    monkeypatch.setattr(pbr, "config_prefilter_metrics_json", forbidden)
    assert config_prefilter({}, None) == 0
    assert (
        config_prefilter(
            {}, {"checks": [{"metric": "grid_span", "min": 0.1, "max": 0.35}]}
        )
        == 0
    )
    assert (
        config_prefilter(
            {},
            {"checks": [{"metric": "entry_close_span", "enabled": False, "max": 0.1}]},
        )
        == 0
    )


def test_scalar_is_pure_graded_and_uses_effective_capacity(
    require_real_passivbot_rust_module,
):
    candidate = config()
    params = policy(max=0.1)
    before = deepcopy((candidate, params))
    first, findings = inspect_config_prefilter(candidate, params)
    assert first == pytest.approx(56000)
    assert findings[0]["value"] == pytest.approx(0.156)
    assert (candidate, params) == before
    candidate["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] = 0.04
    assert config_prefilter(candidate, params) > first
    assert config_prefilter(config(), policy(max=0.2)) == 0


@pytest.mark.parametrize("side", ["long", "short"])
def test_entry_threshold_has_inclusive_bounds_and_graded_penalties(
    require_real_passivbot_rust_module, side
):
    candidate = config()
    candidate["bot"][side] = deepcopy(candidate["bot"]["long"])
    candidate["live"]["approved_coins"] = {"long": [], "short": []}
    candidate["live"]["approved_coins"][side] = ["TEST"]
    strategy = candidate["bot"][side]["strategy"]["trailing_martingale"]
    entry = strategy["entry"]
    entry["threshold_base_pct"] = 0.04
    params = policy("entry_threshold", max=0.2, scale=0.01)
    params["checks"][0]["sides"] = [side]
    assert config_prefilter(candidate, params) == 0
    # A close threshold cannot substitute for the entry-only distance.
    strategy["close"]["threshold_base_pct"] = 0.5
    assert config_prefilter(candidate, params) == 0
    entry["threshold_base_pct"] = 0.042
    value, findings = inspect_config_prefilter(candidate, params)
    assert value == pytest.approx(1e6)
    assert findings[0]["metric"] == "entry_threshold"
    assert findings[0]["value"] == pytest.approx(0.21)
    assert findings[0]["exposure_ratio"] == 0.8
    entry["threshold_base_pct"] = 0.044
    assert config_prefilter(candidate, params) == pytest.approx(2e6)


def test_entry_threshold_uses_worst_exposure_without_multiplying_penalties(
    require_real_passivbot_rust_module,
):
    candidate = config()
    params = policy("entry_threshold", max=0.15)
    # Exactly .15 at 80% utilization; .1725 at 95%.
    assert config_prefilter(candidate, params) == 0
    params["exposure_ratios"] = [0.8, 0.95, 0.95]
    value, findings = inspect_config_prefilter(candidate, params)
    assert value == pytest.approx(22500)
    assert len(findings) == 1
    assert findings[0]["exposure_ratio"] == 0.95


def test_independent_retracement_and_cooldown_checks(
    require_real_passivbot_rust_module,
):
    candidate = config()
    assert config_prefilter(
        candidate, policy("entry_retracement", max=0.001)
    ) == pytest.approx(80)
    assert config_prefilter(
        candidate, policy("close_retracement", max=0.001)
    ) == pytest.approx(1000)
    cooldown = candidate["bot"]["long"]["entry_cooldown"]
    cooldown.update(
        max_duration_minutes=20,
        weights_minutes={"exposure_ratio": 10, "adverse_directionality": 0},
    )
    assert config_prefilter(
        candidate, policy("entry_cooldown_minutes", max=10)
    ) == pytest.approx(3e6)


def test_grid_span_checks_both_bounds_and_incomplete_ladders(
    require_real_passivbot_rust_module,
):
    candidate = config()
    entry = candidate["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]
    entry.update(threshold_we_weight=0, retracement_base_pct=0, initial_ema_dist=0)
    entry["threshold_base_pct"] = 0.0001
    assert config_prefilter(candidate, policy("grid_span", min=0.1, max=0.35)) > 0
    entry["threshold_base_pct"] = 0.05
    assert config_prefilter(candidate, policy("grid_span", min=0.05, max=0.35)) == 0
    entry["threshold_base_pct"] = 0.3
    assert config_prefilter(candidate, policy("grid_span", max=0.1)) > 0
    limited = policy("grid_span", max=100)
    limited["grid_max_entries"] = 2
    value, findings = inspect_config_prefilter(candidate, limited)
    assert value > 0 and findings[0]["grid_incomplete"]


@pytest.mark.parametrize("metric", ["entry_close_span", "entry_threshold"])
def test_coin_and_unscreened_scenario_overrides_are_checked(
    require_real_passivbot_rust_module, metric,
):
    candidate = config()
    candidate["coin_overrides"] = {
        "TEST": {
            "bot": {
                "long": {
                    "strategy": {
                        "trailing_martingale": {
                            "entry": {"threshold_base_pct": 0.001},
                        }
                    }
                }
            }
        }
    }
    params = policy(metric, max=0.1)
    assert config_prefilter(candidate, params) == 0
    stress = config()
    value, findings = prefilter_candidate(
        {"base": candidate, "unscreened": stress}, params
    )
    assert value > 0 and findings[0]["scenario"] == "unscreened"


@pytest.mark.parametrize("side", ["long", "short"])
def test_trailing_grid_completes_on_both_sides(
    require_real_passivbot_rust_module, side
):
    candidate = config()
    candidate["bot"][side] = deepcopy(candidate["bot"]["long"])
    candidate["live"]["approved_coins"] = {"long": [], "short": []}
    candidate["live"]["approved_coins"][side] = ["TEST"]
    entry = candidate["bot"][side]["strategy"]["trailing_martingale"]["entry"]
    entry.update(threshold_base_pct=0.03, threshold_we_weight=0, initial_ema_dist=0)
    params = policy("grid_span", min=100)
    params["checks"][0]["sides"] = [side]
    constraint, findings = inspect_config_prefilter(candidate, params)
    assert constraint > 0
    assert 0 < findings[0]["value"] < 1
    assert not findings[0]["grid_incomplete"]


def test_prepared_venue_aliases_use_the_backtest_override_resolver(
    require_real_passivbot_rust_module,
    tmp_path,
    monkeypatch,
):
    from utils import create_coin_symbol_map_cache

    monkeypatch.chdir(tmp_path)
    create_coin_symbol_map_cache(
        "binance",
        {
            "TEST/USDT:USDT": {
                "id": "TESTUSDT",
                "base": "TEST",
                "swap": True,
                "linear": True,
                "active": True,
            }
        },
        verbose=False,
    )
    candidate = config()
    candidate["coin_overrides"] = {
        "TEST/USDT:USDT": {
            "bot": {
                "long": {
                    "strategy": {
                        "trailing_martingale": {"entry": {"threshold_base_pct": 0.001}},
                    }
                }
            }
        }
    }
    view = prefilter_config_for_exchange(
        candidate,
        {
            "TEST": {"exchange": "binance", "symbol": "TEST/USDT:USDT"},
        },
        "binance",
    )
    assert config_prefilter(view, policy(max=0.1)) == 0


def test_zero_global_wel_derives_capacity_but_zero_coin_wel_disables_entry(
    require_real_passivbot_rust_module,
):
    candidate = config()
    candidate["bot"]["long"]["wallet_exposure_limit"] = 0
    assert config_prefilter(candidate, policy(max=0.1)) > 0
    candidate["bot"]["long"]["wallet_exposure_limit"] = 100
    assert config_prefilter(candidate, policy(max=0.1)) == pytest.approx(56000)
    candidate["coin_overrides"] = {
        "TEST": {"bot": {"long": {"wallet_exposure_limit": 0}}}
    }
    assert config_prefilter(candidate, policy(max=0.1)) == 0


def test_denomination_alias_preserves_its_resolved_override(
    require_real_passivbot_rust_module,
):
    candidate = config()
    candidate["backtest"]["coins"] = {"binance": ["1000ABC"]}
    candidate["live"]["approved_coins"]["long"] = ["ABC"]
    candidate["coin_overrides"] = {
        "1000ABC": {
            "bot": {
                "long": {
                    "strategy": {
                        "trailing_martingale": {"entry": {"threshold_base_pct": 0.001}}
                    }
                }
            }
        }
    }
    view = prefilter_config_for_exchange(candidate, {"1000ABC": {}}, "binance")
    assert config_prefilter(view, policy(max=0.1)) == 0


def test_multiple_venues_combine_check_maxima_before_scenario_selection(
    require_real_passivbot_rust_module,
):
    first, second = config(), config()
    first["bot"]["long"]["entry_cooldown"]["base_duration_minutes"] = 0
    second["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] = 0.001
    params = policy(max=0.1)
    params["checks"].append(
        {
            "metric": "entry_cooldown_minutes",
            "enabled": True,
            "sides": ["long"],
            "max": 1,
        }
    )
    first_constraint = config_prefilter(first, params)
    second_constraint = config_prefilter(second, params)
    combined, findings = prefilter_candidate(
        {"joint": [first, second], "other": first}, params
    )
    assert combined == pytest.approx(first_constraint + second_constraint)
    assert len(findings) == 2
    assert all(row["scenario"] == "joint" for row in findings)


def test_rejected_suite_metrics_are_consumable_by_pareto_reducers():
    from pareto_store import _suite_metrics_to_stats
    from pareto_explorer import _extract_suite_metrics

    base = SimpleNamespace(
        scoring_specs=extract_objective_specs(["adg_strategy_eq"]), limit_checks=[]
    )
    payload = build_prefilter_payload(
        base, [1], 25, [], suite=True, scenario_labels=("base", "stress")
    )
    entry = {"suite_metrics": payload["metrics"]["suite_metrics"]}
    for flat, values in (
        _suite_metrics_to_stats(entry, {"default": "min"}),
        _extract_suite_metrics(entry, {"default": "min"}),
        _extract_suite_metrics(entry, {"default": "mean"}, scenario_labels=["stress"]),
    ):
        assert flat["adg_strategy_eq_mean"] == -25
        assert values["drawdown_worst_strategy_eq"] == 25


def test_single_coin_dynamic_wel_matches_capacity_and_exposure_cooldown(
    require_real_passivbot_rust_module,
):
    candidate = config()
    candidate["bot"]["long"]["risk"].update(n_positions=4, we_excess_allowance_pct=0.5)
    candidate["bot"]["long"]["entry_cooldown"]["weights_minutes"]["exposure_ratio"] = 10
    candidate["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] = 100
    params = policy("entry_cooldown_minutes", max=14)
    # One eligible coin => one effective slot => capped allowance == base WEL.
    assert candidate["backtest"]["dynamic_wel_by_tradability"] is True
    assert config_prefilter(candidate, params) == 0
    candidate["backtest"]["dynamic_wel_by_tradability"] = False
    # Four fixed slots => .25 WEL with .5 excess => exposure ratio 1.2 at 80%.
    assert config_prefilter(candidate, params) == pytest.approx(3e6)


@pytest.mark.parametrize("metric", ["entry_close_span", "entry_threshold"])
def test_ambiguous_dynamic_multicoin_wel_fails_instead_of_guessing(
    require_real_passivbot_rust_module, metric,
):
    candidate = config()
    candidate["bot"]["long"]["risk"]["n_positions"] = 4
    candidate["backtest"]["coins"]["binance"].append("OTHER")
    candidate["live"]["approved_coins"]["long"].append("OTHER")
    assert config_prefilter(candidate, None) == 0
    with pytest.raises(ValueError, match="data-dependent dynamic WEL"):
        config_prefilter(candidate, policy(metric, max=0.2))
    candidate["bot"]["long"]["risk"]["n_positions"] = 1
    assert config_prefilter(candidate, policy(metric, max=0.2)) == 0
    candidate["bot"]["long"]["risk"]["n_positions"] = 4
    # Explicit allocations do not depend on the grow-only tradability count.
    candidate["coin_overrides"] = {
        coin: {"bot": {"long": {"wallet_exposure_limit": 0.25}}}
        for coin in ("TEST", "OTHER")
    }
    assert config_prefilter(candidate, policy(metric, max=0.2)) == 0


@pytest.mark.parametrize("metric", ["grid_span", "entry_threshold"])
def test_defaults_and_clean_roundtrip_keep_explicit_policy(metric):
    candidate = config()
    assert candidate["optimize"]["prefilter"]["checks"] == []
    candidate["optimize"]["prefilter"] = policy(metric, min=0.1, max=0.35)
    assert (
        clean_config(candidate)["optimize"]["prefilter"]
        == candidate["optimize"]["prefilter"]
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"checks": [{"metric": "unknown", "enabled": True, "max": 1}]},
        {"checks": [{"metric": "grid_span", "enabled": 1, "max": 1}]},
        {"checks": [{"metric": "grid_span", "enabled": True}]},
        {"checks": [{"metric": "grid_span", "enabled": True, "min": 2, "max": 1}]},
        {"checks": [{"metric": "grid_span", "scale": 0}]},
        {"exposure_ratios": [1]},
        {"volatility_ema_1m": float("nan")},
        {"grid_max_entries": True},
        {"typo": True},
    ],
)
def test_invalid_policy_is_an_error_not_a_candidate_penalty(bad):
    with pytest.raises((ValueError, TypeError)):
        resolve_prefilter(bad)


def test_penalty_defaults_follow_explicit_objective_directions_without_rescoring():
    specs = extract_objective_specs(
        [
            {"metric": "adg_strategy_eq", "goal": "max"},
            {"metric": "drawdown_worst_strategy_eq", "goal": "min"},
        ]
    )
    base = SimpleNamespace(scoring_specs=specs, limit_checks=[])
    payload = build_prefilter_payload(base, [1, 2], 25, [])
    assert payload["fitness"] == (25, 25)
    assert payload["constraint_violation"] == 25
    assert payload["evaluation_vector"] == [1, 2]
    assert payload["metrics"]["objectives"] == {
        "adg_strategy_eq": -25,
        "drawdown_worst_strategy_eq": 25,
    }
    assert payload["metrics"]["stats"]["fills_count"]["mean"] == -25
    assert payload["metrics"]["stats"]["win_rate"]["mean"] == -25
    assert payload["metrics"]["stats"]["adg_usd"]["mean"] == -25
    assert payload["metrics"]["stats"]["drawdown_worst_btc"]["mean"] == 25
    assert payload["metrics"]["backtest_performed"] is False
