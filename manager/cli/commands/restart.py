from manager.cli.cli import CLICommand
from manager.constants import logger


class Restart(CLICommand):
    doc = """Restart instances that match the arguments."""
    args_optional = ["query"]
    flags = ["-a", "-u", "-s", "-y", "-f"]

    @staticmethod
    def run(cli):
        force = cli.flags.get("force", False)
        silent = cli.flags.get("silent", False)

        instances_to_restart = cli.get_instances_for_action()
        if len(instances_to_restart) == 0:
            return

        if cli.confirm_action("restart", instances_to_restart) != True:
            return

        logger.info("Restarting instances. It may take a while...")
        restarted_instances = []
        failed = []
        progress = cli.add_progress(
            "restarted 0/{}".format(len(instances_to_restart)))
        for instance in instances_to_restart:
            if instance.restart(force, silent):
                restarted_instances.append(instance.get_id())
            else:
                failed.append(instance.get_id())

            progress.update(
                "restarted {}/{}".format(len(restarted_instances), len(instances_to_restart)))

        progress.finish("Restarted {} instance(s)".format(
            len(restarted_instances)))

        if len(failed) > 0:
            logger.info("Failed to restart {} instances:".format(len(failed)))
            for id in failed:
                logger.info("- {}".format(id))
