from __future__ import annotations

import datetime
import os
import pathlib
import shutil
import sys
import tempfile

import nox
from nox.command import CommandFailed

IS_WINDOWS = sys.platform.lower().startswith("win")
IS_DARWIN = sys.platform.lower().startswith("darwin")

if IS_WINDOWS:
    COVERAGE_FAIL_UNDER_PERCENT = 92
elif IS_DARWIN:
    COVERAGE_FAIL_UNDER_PERCENT = 95
else:
    COVERAGE_FAIL_UNDER_PERCENT = 95

# Be verbose when running under a CI context
PIP_INSTALL_SILENT = (
    os.environ.get("JENKINS_URL")
    or os.environ.get("CI")
    or os.environ.get("DRONE")
    or os.environ.get("GITHUB_ACTIONS")
) is None
CI_RUN = PIP_INSTALL_SILENT is False
SKIP_REQUIREMENTS_INSTALL = "SKIP_REQUIREMENTS_INSTALL" in os.environ
EXTRA_REQUIREMENTS_INSTALL = os.environ.get("EXTRA_REQUIREMENTS_INSTALL")

# Paths
REPO_ROOT = pathlib.Path(__file__).resolve().parent
# Change current directory to REPO_ROOT
os.chdir(str(REPO_ROOT))

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
# Make sure the artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SITECUSTOMIZE_DIR = os.path.join(REPO_ROOT, "tests", "support", "coverage")
RUNTESTS_LOGFILE = ARTIFACTS_DIR.relative_to(REPO_ROOT) / "runtests-{}.log".format(
    datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")
)
COVERAGE_REPORT_DB = REPO_ROOT / ".coverage"
COVERAGE_REPORT_PROJECT = ARTIFACTS_DIR.relative_to(REPO_ROOT) / "coverage-project.xml"
COVERAGE_REPORT_TESTS = ARTIFACTS_DIR.relative_to(REPO_ROOT) / "coverage-tests.xml"
JUNIT_REPORT = ARTIFACTS_DIR.relative_to(REPO_ROOT) / "junit-report.xml"

# Nox options
#  Reuse existing virtualenvs
nox.options.reuse_existing_virtualenvs = True
#  Don't fail on missing interpreters
nox.options.error_on_missing_interpreters = False


@nox.session(python=("3", "3.8", "3.9", "3.10"))
def tests(session):
    """
    Run tests
    """
    if SKIP_REQUIREMENTS_INSTALL is False:
        # Always have the wheel package installed
        session.install("wheel", silent=PIP_INSTALL_SILENT)
        pytest_version_requirement = os.environ.get("PYTEST_VERSION_REQUIREMENT") or None
        if pytest_version_requirement:
            if not pytest_version_requirement.startswith("pytest"):
                pytest_version_requirement = f"pytest{pytest_version_requirement}"
            session.install(pytest_version_requirement, silent=PIP_INSTALL_SILENT)
        session.install("-e", ".[dev,tests]", silent=PIP_INSTALL_SILENT)

        if EXTRA_REQUIREMENTS_INSTALL:
            session.log(
                "Installing the following extra requirements because the EXTRA_REQUIREMENTS_INSTALL "
                "environment variable was set: EXTRA_REQUIREMENTS_INSTALL='%s'",
                EXTRA_REQUIREMENTS_INSTALL,
            )
            install_command = ["--progress-bar=off"]
            install_command += [req.strip() for req in EXTRA_REQUIREMENTS_INSTALL.split()]
            session.install(*install_command, silent=PIP_INSTALL_SILENT)

    session.run("coverage", "erase")
    python_path_env_var = os.environ.get("PYTHONPATH") or None
    if python_path_env_var is None:
        python_path_env_var = SITECUSTOMIZE_DIR
    else:
        python_path_entries = python_path_env_var.split(os.pathsep)
        if SITECUSTOMIZE_DIR in python_path_entries:
            python_path_entries.remove(SITECUSTOMIZE_DIR)
        python_path_entries.insert(0, SITECUSTOMIZE_DIR)
        python_path_env_var = os.pathsep.join(python_path_entries)
    env = {
        # The updated python path so that sitecustomize is importable
        "PYTHONPATH": python_path_env_var,
        # The full path to the .coverage data file. Makes sure we always write
        # them to the same directory
        "COVERAGE_FILE": str(COVERAGE_REPORT_DB),
        # Instruct sub processes to also run under coverage
        "COVERAGE_PROCESS_START": os.path.join(REPO_ROOT, ".coveragerc"),
    }
    args = [
        "--rootdir",
        str(REPO_ROOT),
        f"--log-file={RUNTESTS_LOGFILE}",
        "--log-file-level=debug",
        "--show-capture=no",
        f"--junitxml={JUNIT_REPORT}",
        "--showlocals",
        "--strict-markers",
        "-ra",
        "-s",
    ]
    if session._runner.global_config.forcecolor:
        args.append("--color=yes")
    if not session.posargs:
        args.append("tests/")
    else:
        for arg in session.posargs:
            if arg.startswith("--color") and session._runner.global_config.forcecolor:
                args.remove("--color=yes")
            args.append(arg)

    try:
        session.run("coverage", "run", "-m", "pytest", *args, env=env)
    finally:
        try:
            session.run("coverage", "combine")
        except CommandFailed:
            # Coverage data from sub-processes missing
            pass
        # Generate report for project code coverage
        session.run(
            "coverage",
            "xml",
            "-o",
            str(COVERAGE_REPORT_PROJECT),
            "--omit=tests/*",
            "--include=src/passivbot/*",
        )
        # Generate report for tests code coverage
        session.run(
            "coverage",
            "xml",
            "-o",
            str(COVERAGE_REPORT_TESTS),
            "--omit=src/passivbot/*",
            "--include=tests/*",
        )
        try:
            cmdline = [
                "coverage",
                "report",
                "--show-missing",
                "--include=src/passivbot/*,tests/*",
            ]
            if False:
                cmdline.append(f"--fail-under={COVERAGE_FAIL_UNDER_PERCENT}")
            session.run(*cmdline)
        finally:
            if COVERAGE_REPORT_DB.exists():
                shutil.copyfile(str(COVERAGE_REPORT_DB), str(ARTIFACTS_DIR / ".coverage"))


def _lint(session, rcfile, flags, paths):
    session.install(
        "--progress-bar=off",
        "-r",
        os.path.join("requirements", "lint.txt"),
        silent=PIP_INSTALL_SILENT,
    )
    if SKIP_REQUIREMENTS_INSTALL is False:
        # Always have the wheel package installed
        session.install("wheel", silent=PIP_INSTALL_SILENT)
        session.install("-e", ".[dev,tests.lint]", silent=PIP_INSTALL_SILENT)

        if EXTRA_REQUIREMENTS_INSTALL:
            session.log(
                "Installing the following extra requirements because the EXTRA_REQUIREMENTS_INSTALL "
                "environment variable was set: EXTRA_REQUIREMENTS_INSTALL='%s'",
                EXTRA_REQUIREMENTS_INSTALL,
            )
            install_command = ["--progress-bar=off"]
            install_command += [req.strip() for req in EXTRA_REQUIREMENTS_INSTALL.split()]
            session.install(*install_command, silent=PIP_INSTALL_SILENT)
    session.run("pylint", "--version")
    pylint_report_path = os.environ.get("PYLINT_REPORT")

    cmd_args = ["pylint", f"--rcfile={rcfile}"] + list(flags) + list(paths)

    stdout = tempfile.TemporaryFile(mode="w+b")
    try:
        session.run(*cmd_args, stdout=stdout)
    finally:
        stdout.seek(0)
        contents = stdout.read()
        if contents:
            contents = contents.decode("utf-8")
            sys.stdout.write(contents)
            sys.stdout.flush()
            if pylint_report_path:
                # Write report
                with open(pylint_report_path, "w") as wfh:
                    wfh.write(contents)
                session.log("Report file written to %r", pylint_report_path)
        stdout.close()


@nox.session(python="3")
def lint(session):
    """
    Run PyLint against Salt and it's test suite. Set PYLINT_REPORT to a path to capture output.
    """
    session.notify(f"lint-code-{session.python}")
    session.notify(f"lint-tests-{session.python}")


@nox.session(python="3", name="lint-code")
def lint_code(session):
    """
    Run PyLint against the code. Set PYLINT_REPORT to a path to capture output.
    """
    flags = ["--disable=I"]
    if session.posargs:
        paths = session.posargs
    else:
        paths = ["setup.py", "noxfile.py", "src/passivbot/"]
    _lint(session, ".pylintrc", flags, paths)


@nox.session(python="3", name="lint-tests")
def lint_tests(session):
    """
    Run PyLint against Salt and it's test suite. Set PYLINT_REPORT to a path to capture output.
    """
    flags = ["--disable=I"]
    if session.posargs:
        paths = session.posargs
    else:
        paths = ["tests/"]
    _lint(session, ".pylintrc", flags, paths)
