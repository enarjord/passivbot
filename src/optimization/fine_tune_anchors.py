from __future__ import annotations

from typing import Any


ANCHOR_PLAN_KEY = "_fine_tune_anchor_plan"
ANCHOR_GENE_KEY = "__anchor_id__"


def get_anchor_plan(config: dict | None) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return None
    plan = config.get(ANCHOR_PLAN_KEY)
    return plan if isinstance(plan, dict) and plan.get("anchors") else None


def is_anchored_shape_key(key: str) -> bool:
    return key == ANCHOR_GENE_KEY
