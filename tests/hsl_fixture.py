"""Load the canonical synthetic offline HSL fixture."""

from pathlib import Path
from config import prepare_config
import json
from config_utils import clean_config


FAKE_HSL_CONFIG_PATH = Path(__file__).with_name("fixtures") / "fake_live_hsl.json"


def load_fake_hsl_config():
    return prepare_config(
        json.loads(FAKE_HSL_CONFIG_PATH.read_text()), target="canonical", runtime=None, verbose=False
    )


def set_risk_memory_spans(config, span):
    """Set the fixture's HSL and shared realized-loss memory spans."""
    bot = config["bot"]
    for side in ("long", "short"):
        bot[side]["risk"]["hsl"]["memory_ema_span_minutes"] = span
    portfolio_risk = bot["portfolio"]["risk"]
    if portfolio_risk["hsl"]["scope"] == "portfolio":
        portfolio_risk["hsl"]["memory_ema_span_minutes"] = span
    portfolio_risk["loss_realization"]["memory_ema_span_minutes"] = span


def mutable_runtime_fixture(config):
    """Make explicit independent policy for runtime-only fixture mutation.

    Keep the effective example values, including activation, while discarding
    prepared authority before a generated template changes its scopes or ranges.
    Optimizer binding/source-identity tests must use their canonical source and
    reprepare lifecycle instead of this deliberately expanded runtime snapshot.
    """
    source = clean_config(config)
    source["bot"] = json.loads(json.dumps(config["bot"]))
    return source
