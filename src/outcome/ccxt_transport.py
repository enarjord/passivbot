from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from outcome.models import OutcomeVenue


def create_ccxt_prediction_client(
    venue: OutcomeVenue | str,
    config: Mapping[str, Any] | None = None,
    *,
    allow_builder_attribution: bool = False,
) -> Any:
    """Construct a prediction connector without implicit builder attribution.

    CCXT prediction connectors enable their own builder attribution by default.
    Hyperliquid's connector may submit a builder-fee approval before its first
    order, even when the configured builder fee is zero. Passivbot therefore
    disables that behavior unless a caller opts in explicitly.
    """

    try:
        import ccxt.prediction as ccxt_prediction
    except ImportError as exc:  # pragma: no cover - guarded by the live dependency pin
        raise RuntimeError(
            "the installed CCXT build does not expose prediction connectors"
        ) from exc

    normalized_venue = OutcomeVenue(venue)
    client_config = deepcopy(dict(config or {}))
    raw_options = client_config.get("options", {})
    if not isinstance(raw_options, Mapping):
        raise TypeError("CCXT prediction client options must be a mapping")
    options = deepcopy(dict(raw_options))
    if not allow_builder_attribution:
        options["builderFee"] = False
    client_config["options"] = options

    connector = {
        OutcomeVenue.HYPERLIQUID: ccxt_prediction.hyperliquid,
        OutcomeVenue.POLYMARKET: ccxt_prediction.polymarket,
    }[normalized_venue]
    return connector(client_config)
