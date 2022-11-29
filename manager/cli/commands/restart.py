from manager.cli.cli import CLICommand
from manager.constants import logger


class Restart(CLICommand):
    doc = """Restart instances that match the arguments."""
    args_optional = ["query"]
    flags = ["-a", "-u", "-s", "-y", "-f", "-m"]

    @staticmethod
    def run(cli):
        force = cli.flags.get("force", False)
        silent = cli.flags.get("silent", False)

        logger.info("Looking for matching instances...")

        instances_to_restart = cli.get_instances_for_action()
        if len(instances_to_restart) == 0:
            return

        if cli.confirm_action("restart", instances_to_restart) != True:
            return

        logger.info("Restarting instances. It may take a while...")
        restarted_instances = []
        failed = []
        progress = cli.add_progress(f"restarted 0/{len(instances_to_restart)}")
        for instance in instances_to_restart:
            instance.apply_flags(cli.modifiers)
            if instance.restart(force, silent):
                restarted_instances.append(instance.get_id())
            else:
                failed.append(instance.get_id())

            progress.update(
                f"restarted {len(restarted_instances)}/{len(instances_to_restart)}")

        progress.finish(f"Restarted {len(restarted_instances)} instance(s)")

        if len(failed) > 0:
            logger.info(f"Failed to restart {len(failed)} instances:")
            for id in failed:
                logger.info(f"- {id}")
