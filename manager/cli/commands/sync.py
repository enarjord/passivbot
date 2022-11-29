from manager.cli.cli import CLICommand
from manager.constants import logger
from manager.cli.color import Color


class Sync(CLICommand):
    doc = f"""Sync instances with config.
    {Color.apply(Color.YELLOW, "CAUTION:")}
    {Color.apply(Color.UNDERLINE, "all")} instances that are currently running will be
    forcefully stopped, and only the ones that are
    currently in the config will be started again."""

    args_optional = ["query"]
    flags = ["-y", "-s"]

    @staticmethod
    def run(cli):
        cli.flags["all"] = True
        cli.flags["force"] = True

        logger.info(f"{Color.apply(Color.YELLOW, 'CAUTION:')}")
        logger.info(
            "you are about to stop all instances that are currently running.")
        logger.info(
            "Only the instances that are currently in the config file will be started again.")
        if cli.confirm_action("sync", cli.manager.get_instances()) != True:
            return

        cli.flags["yes"] = True

        delimiter = "-" * 30
        logger.info("Syncing instances...")
        logger.info(delimiter)
        cli.run_command("stop")
        logger.info(delimiter)
        cli.manager.load_instances()
        cli.run_command("start")

        logger.info(delimiter)
        logger.info("Sync complete")
        logger.info(delimiter)
        cli.run_command("list")
