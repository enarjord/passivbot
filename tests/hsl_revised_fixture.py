"""Explicit offline migration choices for the repository fake HSL fixture."""
from pathlib import Path
import hjson
from config import prepare_config


def load_fake_hsl_config():
    path = Path(__file__).resolve().parents[1] / "configs/fake_live_hsl_btc.hjson"
    config = hjson.loads(path.read_text())
    config.setdefault("live", {})["hsl_engine"] = "revised"
    config["live"]["pnls_max_lookback_days"] = 1.0
    for side in ("long", "short"):
        config["bot"][side]["hsl_restart_after_red_policy"] = "always"
    return prepare_config(config, target="canonical", runtime=None, verbose=False)
