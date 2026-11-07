from types import SimpleNamespace

import pytest

import procedures


@pytest.fixture(autouse=True)
def enforce_version_check(monkeypatch):
    monkeypatch.delenv("SKIP_CCXT_ASSERT", raising=False)


def test_checkout_uses_its_requirements_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(procedures, "__file__", str(tmp_path / "src" / "procedures.py"))
    (tmp_path / "requirements-live.txt").write_text(
        "# ccxt==0.0.1\nccxt-extra==1.0\nCCXT == 4.5.66 # tested version\n", encoding="utf-8"
    )
    source_metadata = SimpleNamespace(
        locate_file=lambda path: tmp_path / "src" / path,
        read_text=lambda _: None,
        requires=["ccxt==0.0.1"],
    )
    monkeypatch.setattr(procedures.metadata, "distribution", lambda _: source_metadata)

    assert procedures.load_ccxt_version() == "4.5.66"


def test_checkout_missing_requirements_fails_even_with_package_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(procedures, "__file__", str(tmp_path / "src" / "procedures.py"))
    installed = SimpleNamespace(
        locate_file=lambda path: tmp_path / "site-packages" / path,
        read_text=lambda _: "Wheel-Version: 1.0\n",
        requires=["ccxt==4.5.66"],
    )
    monkeypatch.setattr(procedures.metadata, "distribution", lambda _: installed)

    with pytest.raises(FileNotFoundError, match="requirements-live.txt"):
        procedures.load_ccxt_version()


def test_installed_version_is_checked_against_wheel_metadata(tmp_path, monkeypatch):
    (tmp_path / "requirements-live.txt").write_text("ccxt==0.0.1\n", encoding="utf-8")
    site_packages = tmp_path / ".vendor"
    info = site_packages / "passivbot-0.0.0.dist-info"
    info.mkdir(parents=True)
    (info / "WHEEL").write_text("Wheel-Version: 1.0\n", encoding="utf-8")
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: passivbot\nVersion: 0.0.0\n"
        "Requires-Dist: ccxt==4.5.66\nRequires-Dist: ccxt-extra==1.0\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(site_packages))
    monkeypatch.setattr(procedures, "__file__", str(site_packages / "procedures.py"))

    procedures.assert_correct_ccxt_version(ccxt=SimpleNamespace(__version__="4.5.66"))
    with pytest.raises(AssertionError, match=r"install ccxt v4\.5\.66"):
        procedures.assert_correct_ccxt_version(ccxt=SimpleNamespace(__version__="4.5.65"))


@pytest.mark.parametrize(
    "requirements",
    [
        [],
        ["ccxt>=4.5.66"],
        ["ccxt==4.5.*"],
        ["ccxt==4x..!!"],
        ["ccxt==4.5.66; python_version >= '3.12'"],
        ["ccxt==4.5.66", "ccxt>=4.5.65"],
        ["ccxt==4.5.66", "CCXT==4.5.65"],
    ],
)
def test_installed_guard_rejects_missing_or_inexact_pins(tmp_path, monkeypatch, requirements):
    site_packages = tmp_path / "site-packages"
    monkeypatch.setattr(procedures, "__file__", str(site_packages / "procedures.py"))
    installed = SimpleNamespace(
        locate_file=lambda path: site_packages / path,
        read_text=lambda _: "Wheel-Version: 1.0\n",
        requires=requirements,
    )
    monkeypatch.setattr(procedures.metadata, "distribution", lambda _: installed)

    with pytest.raises(RuntimeError, match="one unconditional numeric CCXT version pin"):
        procedures.assert_correct_ccxt_version(ccxt=SimpleNamespace(__version__="4.5.66"))
