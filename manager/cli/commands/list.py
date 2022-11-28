from manager.cli.cli import CLICommand
from manager.constants import logger


class List(CLICommand):
    doc = """List instances that match the arguments.
    List all if no arguments given."""
    args_optional = ["query"]

    @staticmethod
    def run(cli):
        instances_synced = cli.manager.get_synced_instances()
        instances_unsynced = cli.manager.get_unsynced_instances()
        lines = []

        if len(instances_synced) > 0:
            lines.extend(cli.format_instnaces(
                instances_synced, title="Instances"))

        if len(instances_unsynced) > 0:
            lines.extend(cli.format_instnaces(
                instances_unsynced, title="\nUnsynced"))

        lines.append("\n{}".format(cli.manager.count_running(format=True)))
        lines.append("{}".format(cli.manager.count_unsynced(format=True)))
        lines.append(
            '\nUse "manager info" to get more info about a particular instance')

        for line in lines:
            logger.info(line)
