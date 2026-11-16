"""Exercise offline candidate ranking in the same JS runtime as the viewer."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tools.pareto_plot_selectors import SELECTOR_SCRIPT


def test_projection_selectors(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to test the offline selection logic")
    module = tmp_path / "selectors.cjs"
    module.write_text(
        SELECTOR_SCRIPT
        + "\nmodule.exports = {rankProjection, projectionFront, PROJECTION_METHODS};\n",
        encoding="utf-8",
    )
    script = Path(__file__).with_name("pareto_plot_selectors.cjs")
    subprocess.run([node, str(script), str(module)], check=True)
