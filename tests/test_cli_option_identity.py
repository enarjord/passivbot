"""Only explicitly declared long flags and aliases identify a config override."""

import pytest

from cli_utils import add_help_all_argument, build_command_parser


def parser():
    result = build_command_parser(
        prog="passivbot backtest", description="Config flag fixture",
        usage="%(prog)s [options]", epilog="",
    )
    result.add_argument(
        "--bot.long.risk.total_wallet_exposure_limit_ratio", "-ltwel", type=float
    )
    add_help_all_argument(result, help_all=False)
    return result


@pytest.mark.parametrize("flag", [
    "--bot.long.risk.total_wallet_exposure_limit_ratio", "-ltwel",
])
def test_declared_options_and_short_aliases_remain_supported(flag):
    args = vars(parser().parse_args([flag, "1.25"]))
    assert args["bot.long.risk.total_wallet_exposure_limit_ratio"] == 1.25


@pytest.mark.parametrize("flag", [
    "--bot.long.risk.total_wallet_exposure_limit",
    "--bot.long.risk.total_wallet_exposure",
    "--bot.long.risk.total",
])
def test_removed_or_undeclared_prefixes_are_not_config_aliases(flag):
    with pytest.raises(SystemExit) as error:
        parser().parse_args([flag, "1.25"])
    assert error.value.code == 2


def test_full_equals_spelling_and_explicit_help_all_remain_supported():
    args = vars(parser().parse_args([
        "--bot.long.risk.total_wallet_exposure_limit_ratio=1.25", "--help-all",
    ]))
    assert args["bot.long.risk.total_wallet_exposure_limit_ratio"] == 1.25
    assert args["help_all"] is True


def test_parse_known_args_retains_unknown_prefix_for_its_owner():
    args, unknown = parser().parse_known_args([
        "--bot.long.risk.total_wallet_exposure_limit", "1.25",
    ])
    assert vars(args)["bot.long.risk.total_wallet_exposure_limit_ratio"] is None
    assert unknown == ["--bot.long.risk.total_wallet_exposure_limit", "1.25"]
