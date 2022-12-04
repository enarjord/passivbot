from manager.cli.cli import CLICommand
from manager.constants import logger


class List(CLICommand):
    doc = """List instances that match the arguments.
    List all if no arguments given."""
    args_optional = ["query"]

    @staticmethod
    def run(cli):
        instances = cli.get_instances_for_action()
        lines = []

        if len(instances) == 0:
            return

        instances_synced = []
        instances_unsynced = []
        for instance in instances:
            if instance.is_in_config():
                instances_synced.append(instance)
            else:
                instances_unsynced.append(instance)

        if len(instances_synced) > 0:
            lines.extend(cli.format_instnaces(
                instances_synced, title="Instances"))

        if len(instances_unsynced) > 0:
            lines.extend(cli.format_instnaces(
                instances_unsynced, title="\nUnsynced"))

        lines.append(f"\n{cli.manager.count_running(instances, format=True)}")
        lines.append(f"{cli.manager.count_unsynced(instances, format=True)}")
        lines.append(
            '\nUse "manager info" to get more info about a particular instance')

        for line in lines:
            logger.info(line)
