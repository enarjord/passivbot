"""The descriptive registry covers exactly the source-owned core catalog."""

import passivbot_rust as native
import pytest
from config.metric_definitions import METRIC_DEFINITIONS, get_metric_definition
from config.metrics import CORE_METRIC_CATALOG, KNOWN_METRICS
from tools.generate_metric_reference import DEFAULT_OUTPUT, render_reference


def test_registry_matches_native_catalog_and_definition_version():
    catalog = native.get_core_metric_catalog()
    assert catalog["definition_version"] == 1
    assert len(catalog["metrics"]) == 27
    assert METRIC_DEFINITIONS.keys() == KNOWN_METRICS == CORE_METRIC_CATALOG.keys()
    for item in catalog["metrics"]:
        field = get_metric_definition(item["name"])
        assert field.unit == item["unit"]
        assert field.basis and field.sampling and field.aggregation


def test_generated_reference_is_current():
    assert DEFAULT_OUTPUT.read_text() == render_reference()


@pytest.mark.parametrize(
    "name",
    [
        "adg_fraction",
        "sharpe_ratio_usd",
        "hsl_panic_fill_loss_max_usd",
        "long_pnl_to_total_pnl_ratio",
    ],
)
def test_retired_quantities_have_no_descriptive_aliases(name):
    with pytest.raises(KeyError):
        get_metric_definition(name)
