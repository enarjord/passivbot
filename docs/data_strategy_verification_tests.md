# Verification Test Suite: Data Strategy Redesign

You are testing the new flattened scenario configuration system for passivbot's backtest/optimizer suite mode. The key behavior rules are:

| Config State | Expected Behavior |
|--------------|-------------------|
| `scenarios` empty/missing | Single backtest using `backtest.exchanges` |
| `suite_enabled: false` | Single backtest, scenarios ignored |
| `suite_enabled: true` (default) + scenarios | Suite mode |
| Scenario without `exchanges` | Inherits `backtest.exchanges` |
| Scenario with 1 exchange | Data from that exchange only |
| Scenario with N exchanges | Best-per-coin combination |

### CLI Override Behavior

| Config `suite_enabled` | CLI `--suite` | CLI `--scenarios` | Result |
|------------------------|---------------|-------------------|--------|
| `true` (default) | (none) | (none) | Suite mode if scenarios exist |
| `true` | `n` | (none) | Plain backtest, ignore scenarios |
| `false` | (none) | (none) | Plain backtest, ignore scenarios |
| `false` | `y` | (none) | Suite mode |
| any | any | `base,foo` | Suite mode, filtered to matching labels |

For each test case below, verify:
1. Config loads without errors
2. Migration (if legacy config) produces expected structure
3. `build_scenarios()` returns correct scenario list
4. Data preparation uses correct strategy (single vs combined)
5. No runtime errors during execution

---

## Test Categories

### Category 1: Basic Scenario Configurations

#### TC1.1: Empty scenarios (non-suite mode)
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [],  # Empty = single backtest
    }
}
```
**Expected:** Single backtest on binance, NOT suite mode.

#### TC1.2: Single scenario inheriting exchanges
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "scenarios": [
            {"label": "default"}  # No exchanges = inherit
        ]
    }
}
```
**Expected:** Suite mode with 1 scenario using combined data from binance+bybit.

#### TC1.3: Single scenario with explicit single exchange
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "scenarios": [
            {"label": "binance_only", "exchanges": ["binance"]}
        ]
    }
}
```
**Expected:** Suite mode with 1 scenario using binance data only (NOT combined).

#### TC1.4: Single scenario with explicit multiple exchanges
```python
config = {
    "backtest": {
        "exchanges": ["binance"],  # Base has 1
        "scenarios": [
            {"label": "combined", "exchanges": ["binance", "bybit"]}  # Override to 2
        ]
    }
}
```
**Expected:** Suite mode with 1 scenario using combined binance+bybit data.

#### TC1.5: Multiple scenarios, all inheriting
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "scenarios": [
            {"label": "scenario_a"},
            {"label": "scenario_b"},
            {"label": "scenario_c"}
        ]
    }
}
```
**Expected:** 3 scenarios, each using combined binance+bybit data.

#### TC1.6: Multiple scenarios with different single exchanges
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "scenarios": [
            {"label": "binance", "exchanges": ["binance"]},
            {"label": "bybit", "exchanges": ["bybit"]}
        ]
    }
}
```
**Expected:** 2 scenarios - first uses binance only, second uses bybit only. This was the original failing case that motivated the redesign.

#### TC1.7: Mixed inheritance patterns
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "scenarios": [
            {"label": "combined"},  # Inherits = combined
            {"label": "binance_only", "exchanges": ["binance"]},  # Single
            {"label": "bybit_only", "exchanges": ["bybit"]},  # Single
            {"label": "explicit_combined", "exchanges": ["binance", "bybit"]}  # Explicit combined
        ]
    }
}
```
**Expected:** 4 scenarios with correct data strategies for each.

---

### Category 2: Exchange Variations

#### TC2.1: Base has single exchange, scenarios inherit
```python
config = {
    "backtest": {
        "exchanges": ["binance"],  # Single exchange base
        "scenarios": [
            {"label": "default"}
        ]
    }
}
```
**Expected:** Single exchange mode (NOT combined), even though it's suite mode.

#### TC2.2: Scenario overrides with fewer exchanges
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit", "kucoin"],
        "scenarios": [
            {"label": "two_only", "exchanges": ["binance", "bybit"]}
        ]
    }
}
```
**Expected:** Combined mode with only binance+bybit (kucoin ignored).

#### TC2.3: Scenario overrides with completely different exchange
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "scenarios": [
            {"label": "kucoin_only", "exchanges": ["kucoin"]}
        ]
    }
}
```
**Expected:** Single exchange mode using kucoin (even though not in base).

#### TC2.4: Multiple scenarios each picking different subset
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit", "kucoin", "bitget"],
        "scenarios": [
            {"label": "tier1", "exchanges": ["binance", "bybit"]},
            {"label": "tier2", "exchanges": ["kucoin", "bitget"]},
            {"label": "all", "exchanges": ["binance", "bybit", "kucoin", "bitget"]}
        ]
    }
}
```
**Expected:** 3 scenarios with different exchange combinations.

---

### Category 3: Coin Variations

#### TC3.1: Scenario with explicit coin list
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "btc_eth", "coins": ["BTC", "ETH"]}
        ]
    },
    "live": {
        "approved_coins": ["BTC", "ETH", "SOL", "ADA"]
    }
}
```
**Expected:** Scenario uses only BTC and ETH.

#### TC3.2: Scenario with ignored_coins
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "no_memes", "ignored_coins": ["DOGE", "SHIB"]}
        ]
    },
    "live": {
        "approved_coins": ["BTC", "ETH", "DOGE", "SHIB"]
    }
}
```
**Expected:** Scenario uses BTC and ETH only (DOGE, SHIB ignored).

#### TC3.3: Scenarios with overlapping coin lists
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "set_a", "coins": ["BTC", "ETH", "SOL"]},
            {"label": "set_b", "coins": ["ETH", "SOL", "ADA"]}
        ]
    }
}
```
**Expected:** Master coin set = BTC, ETH, SOL, ADA. Each scenario filters appropriately.

#### TC3.4: Scenarios with disjoint coin lists
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "majors", "coins": ["BTC", "ETH"]},
            {"label": "alts", "coins": ["SOL", "ADA"]}
        ]
    }
}
```
**Expected:** No overlap required - each scenario works independently.

#### TC3.5: Scenario with coin_sources (per-coin exchange assignment)
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "scenarios": [
            {
                "label": "mixed_sources",
                "coin_sources": {"BTC": "binance", "ETH": "bybit"}
            }
        ]
    }
}
```
**Expected:** BTC data from binance, ETH data from bybit.

#### TC3.6: Conflicting coin_sources across scenarios (should error)
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "scenarios": [
            {"label": "a", "coin_sources": {"BTC": "binance"}},
            {"label": "b", "coin_sources": {"BTC": "bybit"}}  # Conflict!
        ]
    }
}
```
**Expected:** Error raised about conflicting coin_sources (message includes "forces" indicating the conflict).

---

### Category 4: Date/Time Variations

#### TC4.1: Scenario with custom start_date
```python
config = {
    "backtest": {
        "start_date": "2021-01-01",
        "end_date": "2024-01-01",
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "recent", "start_date": "2023-01-01"}
        ]
    }
}
```
**Expected:** Scenario uses 2023-01-01 to 2024-01-01.

#### TC4.2: Scenario with custom end_date
```python
config = {
    "backtest": {
        "start_date": "2021-01-01",
        "end_date": "2024-01-01",
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "early", "end_date": "2022-01-01"}
        ]
    }
}
```
**Expected:** Scenario uses 2021-01-01 to 2022-01-01.

#### TC4.3: Scenarios with different date windows
```python
config = {
    "backtest": {
        "start_date": "2021-01-01",
        "end_date": "2024-01-01",
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "2021", "start_date": "2021-01-01", "end_date": "2022-01-01"},
            {"label": "2022", "start_date": "2022-01-01", "end_date": "2023-01-01"},
            {"label": "2023", "start_date": "2023-01-01", "end_date": "2024-01-01"}
        ]
    }
}
```
**Expected:** Each scenario uses its own date window. Master dataset covers full range.

---

### Category 5: Override Variations

#### TC5.1: Scenario with bot parameter overrides
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [
            {
                "label": "conservative",
                "overrides": {
                    "bot.long.total_wallet_exposure_limit": 0.5,
                    "bot.short.total_wallet_exposure_limit": 0.5
                }
            }
        ]
    },
    "bot": {
        "long": {"total_wallet_exposure_limit": 1.0},
        "short": {"total_wallet_exposure_limit": 1.0}
    }
}
```
**Expected:** Scenario uses TWEL=0.5 instead of 1.0.

#### TC5.2: Scenario disabling a position side
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [
            {
                "label": "long_only",
                "overrides": {"bot.short.total_wallet_exposure_limit": 0}
            },
            {
                "label": "short_only",
                "overrides": {"bot.long.total_wallet_exposure_limit": 0}
            }
        ]
    }
}
```
**Expected:** First scenario is long-only, second is short-only.

#### TC5.3: Scenario with grid/trailing ratio overrides
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [
            {
                "label": "pure_grid",
                "overrides": {
                    "bot.long.entry_trailing_grid_ratio": 0,
                    "bot.long.close_trailing_grid_ratio": 0
                }
            },
            {
                "label": "pure_trailing",
                "overrides": {
                    "bot.long.entry_trailing_grid_ratio": 1,
                    "bot.long.close_trailing_grid_ratio": 1
                }
            }
        ]
    }
}
```
**Expected:** First scenario uses pure grid, second uses pure trailing.

#### TC5.4: Scenario with n_positions override
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "n3", "overrides": {"bot.long.n_positions": 3}},
            {"label": "n5", "overrides": {"bot.long.n_positions": 5}},
            {"label": "n10", "overrides": {"bot.long.n_positions": 10}}
        ]
    }
}
```
**Expected:** Each scenario uses different n_positions.

---

### Category 6: Aggregation Variations

#### TC6.1: Default aggregation (mean)
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "aggregate": {"default": "mean"},
        "scenarios": [
            {"label": "a"},
            {"label": "b"}
        ]
    }
}
```
**Expected:** Suite metrics aggregated using mean.

#### TC6.2: Custom aggregation mode
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "aggregate": {"default": "min"},
        "scenarios": [
            {"label": "a"},
            {"label": "b"}
        ]
    }
}
```
**Expected:** Suite metrics aggregated using min (worst-case).

#### TC6.3: Per-metric aggregation
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "aggregate": {
            "default": "mean",
            "mdg": "min",
            "sharpe_ratio": "median"
        },
        "scenarios": [
            {"label": "a"},
            {"label": "b"}
        ]
    }
}
```
**Expected:** mdg uses min, sharpe uses median, others use mean.

---

### Category 7: Legacy Config Migration

#### TC7.1: Old config with suite.enabled=true
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "suite": {
            "enabled": True,
            "scenarios": [{"label": "test"}],
            "aggregate": {"default": "mean"}
        }
    }
}
```
**Expected:** Migrates to `scenarios` at top level, `suite` wrapper removed.

#### TC7.2: Old config with include_base_scenario=true
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "suite": {
            "enabled": True,
            "include_base_scenario": True,
            "base_label": "my_base",
            "scenarios": [{"label": "custom"}]
        }
    }
}
```
**Expected:** Migrated scenarios = [{"label": "my_base"}, {"label": "custom"}].

#### TC7.3: Old config with include_base_scenario=false
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "suite": {
            "enabled": True,
            "include_base_scenario": False,
            "scenarios": [{"label": "custom"}]
        }
    }
}
```
**Expected:** Migrated scenarios = [{"label": "custom"}] (no base prepended).

#### TC7.4: Old config with combine_ohlcvs=true
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "combine_ohlcvs": True
    }
}
```
**Expected:** `combine_ohlcvs` removed, behavior derived from exchange count (multi = combined).

#### TC7.5: Old config with combine_ohlcvs=false
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "combine_ohlcvs": False
    }
}
```
**Expected:** `combine_ohlcvs` removed. With 2 exchanges and no scenarios, still uses combined (new behavior).

#### TC7.6: Full legacy migration
```python
config = {
    "backtest": {
        "exchanges": ["binance", "bybit"],
        "combine_ohlcvs": True,
        "suite": {
            "enabled": True,
            "include_base_scenario": True,
            "base_label": "combined",
            "aggregate": {"default": "mean", "mdg": "min"},
            "scenarios": [
                {"label": "binance", "exchanges": ["binance"]},
                {"label": "bybit", "exchanges": ["bybit"]}
            ]
        }
    }
}
```
**Expected:**
- `combine_ohlcvs` removed
- `suite` removed
- `scenarios` = [{"label": "combined"}, {"label": "binance", ...}, {"label": "bybit", ...}]
- `aggregate` = {"default": "mean", "mdg": "min"}

---

### Category 8: Edge Cases and Error Handling

#### TC8.1: Empty exchanges with scenarios (should work if scenarios have exchanges)
```python
config = {
    "backtest": {
        "exchanges": [],
        "scenarios": [
            {"label": "binance", "exchanges": ["binance"]}
        ]
    }
}
```
**Expected:** Works - scenario provides exchanges.

#### TC8.2: Empty exchanges with inheriting scenario (should error)
```python
config = {
    "backtest": {
        "exchanges": [],
        "scenarios": [
            {"label": "default"}  # No exchanges to inherit
        ]
    }
}
```
**Expected:** Error - no exchanges available.

#### TC8.3: Scenario with invalid exchange name
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "bad", "exchanges": ["not_an_exchange"]}
        ]
    }
}
```
**Expected:** Error during data preparation (exchange not found).

#### TC8.4: Very large number of scenarios
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "scenarios": [{"label": f"s{i}"} for i in range(50)]
    }
}
```
**Expected:** All 50 scenarios execute (stress test).

#### TC8.5: Scenario with overlapping but valid date ranges
```python
config = {
    "backtest": {
        "start_date": "2023-01-01",
        "end_date": "2023-06-01",
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "q1", "start_date": "2023-01-01", "end_date": "2023-04-01"},
            {"label": "q2", "start_date": "2023-03-01", "end_date": "2023-06-01"}
        ]
    }
}
```
**Expected:** Works - overlapping date ranges are allowed.

---

### Category 9: Integration Tests

#### TC9.1: Backtest CLI with scenarios
```bash
python src/backtest.py config_with_scenarios.json
```
**Expected:** Suite mode detected, all scenarios executed, summary produced.

#### TC9.2: Backtest CLI with --suite flag
```bash
python src/backtest.py config.json --suite
```
**Expected:** Suite mode forced on even if scenarios might be empty.

#### TC9.3: Optimizer with scenarios
```bash
python src/optimize.py config_with_scenarios.json
```
**Expected:** Suite mode detected, candidates evaluated across all scenarios.

#### TC9.4: Suite config override file
```bash
python src/backtest.py base_config.json --suite-config override_scenarios.json
```
**Expected:** Scenarios from override file used instead of base.

#### TC9.5: Filter scenarios with --scenarios
```bash
python src/backtest.py config.json --scenarios base
python src/backtest.py config.json -sc base,binance_only
```
**Expected:** Only scenarios matching the labels run. Implies `--suite y`.

#### TC9.6: Disable suite with --suite n
```bash
python src/backtest.py config.json --suite n
```
**Expected:** Plain backtest, scenarios ignored even if defined in config.

#### TC9.7: Config with suite_enabled=false
```python
config = {
    "backtest": {
        "exchanges": ["binance"],
        "suite_enabled": False,  # Disable suite by default
        "scenarios": [{"label": "test"}]
    }
}
```
```bash
python src/backtest.py config.json        # Plain backtest (config disables)
python src/backtest.py config.json --suite y  # Suite mode (CLI overrides)
```
**Expected:** Config `suite_enabled` controls default, CLI `--suite` overrides.

#### TC9.8: --scenarios with no matches (error)
```bash
python src/backtest.py config.json --scenarios nonexistent_label
```
**Expected:** Error with message listing available scenario labels.

---

## Verification Checklist

For each test case, verify:

- [ ] Config loads without error (`load_config`, `format_config`)
- [ ] Migration produces expected structure (check `backtest.scenarios`, `backtest.aggregate`)
- [ ] Legacy keys removed (`suite`, `combine_ohlcvs`)
- [ ] `build_scenarios()` returns correct count and labels
- [ ] Exchange inheritance works as expected
- [ ] Data preparation strategy matches exchange count
- [ ] Scenario filtering (coins, dates) applies correctly
- [ ] Overrides apply to scenario config
- [ ] Aggregation uses correct mode
- [ ] No runtime errors during execution

---

## Files to Test

| File | Test Focus |
|------|------------|
| `src/config_utils.py` | Migration, validation |
| `src/suite_runner.py` | `build_scenarios()`, `extract_suite_config()`, data prep |
| `src/backtest.py` | CLI routing, suite mode detection |
| `src/optimize.py` | CLI routing, suite mode detection |
| `src/optimize_suite.py` | `ensure_suite_config()`, context preparation |

---

## How to Run Verification

1. **Unit tests:** `pytest tests/test_suite_runner.py tests/test_config_optimize_suite.py tests/test_suite_config_sources.py -v`

2. **Manual config tests:** Create test configs for each category and run through `format_config()` to verify migration.

3. **Integration tests:** Run actual backtests with small date ranges and coin lists to verify end-to-end behavior.

```python
# Quick validation script
from config_utils import load_config, format_config, get_template_config
from suite_runner import build_scenarios, extract_suite_config

def verify_config(config_dict, expected_scenarios, expected_exchanges_per_scenario):
    formatted = format_config(config_dict, verbose=False)
    suite_cfg = extract_suite_config(formatted, None)

    if suite_cfg.get("enabled"):
        scenarios, aggregate = build_scenarios(suite_cfg, formatted.get("backtest", {}).get("exchanges", []))
        assert len(scenarios) == expected_scenarios
        for i, expected_ex in enumerate(expected_exchanges_per_scenario):
            actual = scenarios[i].exchanges or formatted["backtest"]["exchanges"]
            assert actual == expected_ex, f"Scenario {i}: expected {expected_ex}, got {actual}"

    print(f"✓ Config verified: {len(scenarios) if suite_cfg.get('enabled') else 0} scenarios")
```
