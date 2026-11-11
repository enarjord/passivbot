from pathlib import Path
import shutil
import subprocess

import pytest

from pareto_explorer import ParetoCandidate
from tools.pareto_plot_correlations import CORRELATION_SCRIPT, build_parameter_columns


def test_parameter_columns_include_only_varying_numeric_bot_values():
    candidates = [
        ParetoCandidate(
            path=Path(f"{i}.json"),
            entry={
                "bot": {
                    "long": {
                        "exposure": i,
                        "constant": 4,
                        "flag": i % 2 == 0,
                        "sparse": i if i < 2 else None,
                        "undefined": float("inf"),
                    },
                    **({"short": {"entry": [i]}} if i else {}),
                },
                "coin_overrides": {"TEST": {"bot": {"long": {"exposure": 2 * i}}, "live": {"leverage": i}}},
                "live": {"leverage": i},
                "backtest": {"starting_balance": i},
            },
            objectives={},
            stats_flat={},
            aggregated_values={},
        )
        for i in range(4)
    ]

    assert build_parameter_columns(candidates) == [
        {"key": "bot.long.exposure", "values": [0.0, 1.0, 2.0, 3.0]},
        {"key": "bot.short.entry.0", "values": [None, 1.0, 2.0, 3.0]},
        {"key": "coin_overrides.TEST.bot.long.exposure", "values": [0.0, 2.0, 4.0, 6.0]},
    ]
    assert build_parameter_columns(candidates[:2]) == []


def test_browser_correlations_use_complete_filtered_pairs_and_average_rank_ties():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for browser logic checks")
    program = CORRELATION_SCRIPT + """
const assert = require("node:assert/strict");
const parameters = [
    {key:"positive", values:[0, 1, 2, 3, 4, 5]},
    {key:"negative", values:[3, null, 1, 0, -1, -2]},
    {key:"constant", values:[3, 3, 3, 3, 3, 3]},
    {key:"sparse", values:[3, 3, 4, null, null, 5]},
];
const metric = {values:[0, 1, null, 3, 4, 100]};
const rows = [0, 1, 2, 3, 4];
assert.deepEqual(completeParameterPairs(parameters[1], metric, rows), {
    x:[3, 0, -1], y:[0, 3, 4], indices:[0, 3, 4],
});
assert.deepEqual(parameterMetricCorrelations(parameters, metric, rows, "pearson").map(
    ({key, count}) => [key, count]), [["negative", 3], ["positive", 4]]);
const ranked = parameterMetricCorrelations(parameters, metric, rows, "spearman");
assert.ok(Math.abs(ranked.find(row => row.key === "positive").corr - 1) < 1e-14);
assert.ok(Math.abs(ranked.find(row => row.key === "negative").corr + 1) < 1e-14);
assert.deepEqual(parameterMetricCorrelations(parameters, metric, [0, 1], "pearson"), []);
assert.deepEqual(averageRanks([10, 10, 40, 30]), [1.5, 1.5, 4, 3]);
assert.ok(Math.abs(pearsonCorrelation([-1e308, 0, 1e308], [10, 20, 30]) - 1) < 1e-14);
assert.equal(pearsonCorrelation([10, 10, 10], [3, 4, 5]), null);
// A nonlinear monotonic association keeps rank correlation 1.
const nonlinear = [{key:"param", values:[1, 2, 3, 4]}];
const curve = {values:[1, 2, 4, 32]};
assert.ok(parameterMetricCorrelations(nonlinear, curve, [0, 1, 2, 3], "pearson")[0].corr < 0.9);
assert.ok(Math.abs(parameterMetricCorrelations(nonlinear, curve, [0, 1, 2, 3], "spearman")[0].corr - 1) < 1e-14);
"""
    result = subprocess.run([node, "-e", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
