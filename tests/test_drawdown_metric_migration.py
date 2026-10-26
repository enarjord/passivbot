"""Released-v8 conversions require explicit consent to drawdown formula changes."""
from copy import deepcopy

import pytest

from config.migrations.final_v9 import replace_retired_metrics


@pytest.mark.parametrize('old,new', [
    ('strategy_eq_underwater_pct_mean', 'pain_index_fraction_strategy_equity'),
    ('strategy_eq_underwater_pct_median', 'ulcer_index_fraction_strategy_equity'),
])
def test_old_drawdown_formula_requires_explicit_replacement_and_acknowledgment(old, new):
    source = {'optimize': {'scoring': [{'metric': old, 'goal': 'min'}]}}
    before = deepcopy(source)
    with pytest.raises(ValueError, match='requires --metric-replacement'):
        replace_retired_metrics(source, replacements={}, acknowledge=False, changes=[])
    assert source == before
    with pytest.raises(ValueError, match='acknowledge-metric-formula-change'):
        replace_retired_metrics(source, replacements={old: new}, acknowledge=False, changes=[])
    assert source == before
    changes = []
    replace_retired_metrics(source, replacements={old: new}, acknowledge=True, changes=changes)
    assert source['optimize']['scoring'] == [{'metric': new, 'goal': 'min'}]
    assert changes[0]['reason'] == 'acknowledged explicit metric formula replacement'
