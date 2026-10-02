"""Load the canonical synthetic offline HSL example."""

from pathlib import Path
from config import prepare_config
import json


def load_fake_hsl_config():
    path = Path(__file__).resolve().parents[1] / "configs/examples/fake_live_hsl.json"
    return prepare_config(
        json.loads(path.read_text()), target="canonical", runtime=None, verbose=False
    )
