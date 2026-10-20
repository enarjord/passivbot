"""Load the canonical synthetic offline HSL example."""

from pathlib import Path
from config import prepare_config
import json


def load_fake_hsl_config():
    path = Path(__file__).resolve().parents[1] / "configs/examples/fake_live_hsl.json"
    return prepare_config(
        json.loads(path.read_text()), target="canonical", runtime=None, verbose=False
    )


def set_risk_memory_spans(config, span):
    """Author each independent policy with a common test-fixture value."""
    for side in ("long", "short"):
        config["bot"][side]["hsl"]["memory_ema_span_minutes"] = span
    if config["bot"]["global"]["hsl"]["scope"] == "portfolio":
        config["bot"]["global"]["hsl"]["memory_ema_span_minutes"] = span
    config["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] = span
    config["bot"]["global"]["loss_gate"]["memory_ema_span_minutes"] = span
