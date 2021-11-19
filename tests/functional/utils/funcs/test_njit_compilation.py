import logging
import os
import subprocess
import sys
import textwrap

import pytest

from tests.support.processes import ProcessResult

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ["envvar_value", "argv", "njit_compiled"],
    (
        (None, (), True),
        ("False", (), True),
        ("0", (), True),
        ("true", (), False),
        ("1", (), False),
        (None, ("--nojit",), False),
    ),
)
def test_njit_compilation_by_env_var(envvar_value, argv, njit_compiled, tmp_path):
    code = textwrap.dedent(
        """\
    import sys
    import logging
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    import passivbot.utils.funcs.njit
    logging.getLogger("numba").setLevel(logging.ERROR)

    print(passivbot.utils.funcs.njit.round_dynamic(3.214, 1), file=sys.stdout, flush=True)
    """
    )
    source_path = tmp_path / "njit-test-code.py"
    source_path.write_text(code)
    env = os.environ.copy()
    if envvar_value is not None:
        env["NOJIT"] = envvar_value
    proc = subprocess.run(
        [sys.executable, str(source_path)] + list(argv),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    ret = ProcessResult(
        proc.returncode, proc.stdout.rstrip(), proc.stderr.rstrip(), cmdline=proc.args
    )
    log.debug(ret)
    assert ret.exitcode == 0
    assert ret.stdout.strip() == "3.0"
    if njit_compiled:
        assert "numba.njit compilation is enabled" in ret.stderr
    else:
        assert "numba.njit compilation is disabled" in ret.stderr
