"""Node-backed checks for profile scaling and trace construction."""

import shutil
import subprocess

import pytest

from tools.pareto_plot_tradeoffs import TRADEOFF_SCRIPT


def test_metric_profiles_use_finite_filtered_ranges_and_preserve_gaps(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for browser logic tests")
    script = tmp_path / "profiles.cjs"
    script.write_text(
        TRADEOFF_SCRIPT
        + r"""
const assert = require("node:assert/strict");
const finite = Number.isFinite;
const escapeHtml = text => text.replaceAll("<", "&lt;");
const candidateLabel = (selection, filename) => filename;
const formatValue = String;
const metrics = [
    {key:"growth", label:"growth", values:[-1, 0.5, 0.3, 1, null]},
    {key:"risk", label:"risk", values:[0.1, 0.8, 0.2, null, Infinity]},
    {key:"recovery", label:"recovery", values:[0, 1, 1, 1, NaN]},
];
const dataset = {names:["outside", "a<.json", "b.json", "c.json", "missing"]};
const directions = new Map([["growth","max"], ["risk","min"]]);
const profiles = tradeoffProfiles(metrics, [1,2,3,4], directions);
assert.deepEqual(profiles.rows, [1,2,3]);
assert.equal(profiles.complete, 2);
assert.deepEqual(profiles.ranges.map(({low, high}) => [low, high]), [[0.3,1],[0.2,0.8],[1,1]]);
assert.equal(profilePosition(1, profiles.ranges[0]), 1);
assert.equal(profilePosition(0.2, profiles.ranges[1]), 1);
assert.equal(profilePosition(0.8, profiles.ranges[1]), 0);
assert.equal(profilePosition(1, profiles.ranges[2]), 0.5);
assert.equal(profilePosition(null, profiles.ranges[1]), null);
assert.equal(profilePosition(Infinity, profiles.ranges[1]), null);
assert.equal(profilePosition(0, {low:-1e308, high:1e308}), 0.5);
assert.equal(profilePosition(1, {low:0, high:2}), 0.5);

const {traces} = profileChart(dataset, profiles, 1, 3);
assert.equal(traces.length, 3);
assert.equal(traces[0].connectgaps, false);
assert.equal(traces[0].y[9], null);
assert.deepEqual(traces[1].customdata[0], [1,"a&lt;.json","growth","0.5"]);
assert.equal(traces[2].y[1], null);
assert.equal(profileChart(dataset, profiles, 0, 1).traces.length, 2);
assert.equal(profileChart(dataset, profiles, 1, 1).traces.length, 2);
assert.deepEqual(tradeoffProfiles(metrics, [], directions).rows, []);
""",
        encoding="utf-8",
    )
    subprocess.run([node, str(script)], check=True, capture_output=True, text=True)
