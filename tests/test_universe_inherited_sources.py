"""Prepared source membership retains declaration presence outside JSON policy."""

from copy import deepcopy
import json

import pytest

from config.universe import ResolvedCoinList


@pytest.mark.parametrize("present", [False, True])
def test_prepared_source_presence_survives_copy_but_is_not_json_authority(present):
    value = ResolvedCoinList(["BTC"], None, ["BTC"], source_present=present)
    copied = deepcopy(value)
    assert copied.source_present is present
    assert copied.source is None
    assert copied.shared_source == ["BTC"]
    assert json.loads(json.dumps(value)) == ["BTC"]
    assert type(json.loads(json.dumps(value))) is list


@pytest.mark.parametrize("present", [None, 0, 1, "false", []])
def test_prepared_source_presence_requires_an_exact_internal_boolean(present):
    with pytest.raises(TypeError):
        ResolvedCoinList(["BTC"], None, ["BTC"], source_present=present)
