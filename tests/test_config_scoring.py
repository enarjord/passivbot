from config.scoring import default_objective_goal, normalize_scoring_entries


def test_default_objective_goal_recognizes_new_ratio_metrics():
    assert default_objective_goal("paper_loss_ratio") == "max"
    assert default_objective_goal("paper_loss_mean_ratio") == "max"
    assert default_objective_goal("exposure_ratio") == "max"
    assert default_objective_goal("exposure_mean_ratio") == "max"
    assert default_objective_goal("paper_loss_ratio_usd") == "max"
    assert default_objective_goal("exposure_ratio_btc") == "max"


def test_normalize_scoring_entries_accepts_new_ratio_metrics():
    specs, changed = normalize_scoring_entries(
        ["paper_loss_ratio", "paper_loss_ratio_w", "exposure_ratio", "exposure_mean_ratio_w"]
    )

    assert changed
    assert [(spec.metric, spec.goal) for spec in specs] == [
        ("paper_loss_ratio_usd", "max"),
        ("paper_loss_ratio_w_usd", "max"),
        ("exposure_ratio_usd", "max"),
        ("exposure_mean_ratio_w_usd", "max"),
    ]
