from manager.cli.cli import CLICommand
from manager.constants import logger


class Start(CLICommand):
    doc = """Start instances that match the arguments."""
    args_optional = ["query"]
    flags = ["-a", "-s", "-y", "-m"]

    @staticmethod
    def run(cli):
        silent = cli.flags.get("silent", False)

        logger.info("Looking for stopped instances...")

        instances_to_start = cli.get_instances_for_action(
            lambda i: not i.is_running())
        if len(instances_to_start) == 0:
            return

        if cli.confirm_action("start", instances_to_start) != True:
            return

        logger.info("Starting instances...")
        started_instances = []
        failed = []
        for instance in instances_to_start:
            instance.apply_flags(cli.modifiers)
            started = instance.start(silent)
            if started:
                started_instances.append(instance.get_id())
            else:
                failed.append(instance.get_id())

        logger.info(f"Started {len(started_instances)} instance(s)")

        if len(failed) > 0:
            logger.info(f"Failed to start {len(failed)} instances:")
            for id in failed:
                logger.info(f"- {id}")
