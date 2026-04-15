import json
from pathlib import Path

import pytest

from ccxt_contracts import summarize_market_snapshot


FIXTURE_ROOT = Path("tests/fixtures/ccxt_contracts")


def _fixture_paths():
    if not FIXTURE_ROOT.exists():
        return []
    return sorted(FIXTURE_ROOT.rglob("*.json"))


@pytest.mark.parametrize("fixture_path", _fixture_paths())
def test_captured_market_fixtures_replay_current_contract_summary(fixture_path):
    snapshot = json.loads(fixture_path.read_text())
    if "markets" not in snapshot or "summary" not in snapshot["markets"]:
        pytest.skip("fixture missing market summary")

    exchange = snapshot["meta"]["exchange"]
    quote = snapshot["meta"]["quote"]
    replay = summarize_market_snapshot(exchange, quote, snapshot["markets"]["raw"])

    assert replay["contracts"] == snapshot["markets"]["summary"]["contracts"]
    assert replay["coin_to_symbol_map"] == snapshot["markets"]["summary"]["coin_to_symbol_map"]
    assert replay["eligible_symbols"] == snapshot["markets"]["summary"]["eligible_symbols"]
    assert replay["ineligible_symbols"] == snapshot["markets"]["summary"]["ineligible_symbols"]
    assert replay["ineligible_reasons"] == snapshot["markets"]["summary"]["ineligible_reasons"]
    assert replay["symbol_to_coin_map"] == snapshot["markets"]["summary"]["symbol_to_coin_map"]
