"""Current balance validation remains strict independently of HSL history."""
import pytest
from live.balance_validation import RiskInputUnavailable, validate_balances

@pytest.mark.parametrize('raw,sizing', [(1.,1.),(.01,100.),(100.,.01)])
def test_valid_current_balances(raw,sizing):
    validate_balances(raw,sizing)

@pytest.mark.parametrize('value', [0.,-1.,float('nan'),float('inf'),float('-inf')])
@pytest.mark.parametrize('field', ['raw','sizing'])
def test_invalid_current_balance_is_unavailable(value,field):
    args={'raw':1.,'sizing':1.};args[field]=value
    with pytest.raises(RiskInputUnavailable, match='current_balance_unavailable') as error:
        validate_balances(**args)
    assert error.value.details['balance_raw'] is None or error.value.details['balance_raw']==args['raw']
