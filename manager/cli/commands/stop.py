from manager.cli.cli import CLICommand
from manager.constants import logger


class Stop(CLICommand):
    doc = """Start instances that match the arguments."""
    args_optional = ["query"]
    flags = ["-a", "-u", "-s", "-y", "-f"]

    @staticmethod
    def run(cli):
        force = cli.flags.get("force", False)

        logger.info("Seeking for running instances...")

        instances_to_stop = cli.get_instances_for_action(
            lambda i: i.is_running())
        if len(instances_to_stop) == 0:
            return

        if cli.confirm_action("stop", instances_to_stop) != True:
            return

        logger.info("Stopping instances. This may take a while...")
        stopped_instances = []
        failed = []
        for instance in instances_to_stop:
            if instance.stop(force):
                stopped_instances.append(instance.get_id())
            else:
                failed.append(instance.get_id())

        logger.info("Stopped {} instance(s)".format(len(stopped_instances)))
        if len(failed) > 0:
            logger.info("Failed to stop {} instances:".format(len(failed)))
            for id in failed:
                logger.info("- {}".format(id))
