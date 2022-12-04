from manager.cli.cli import CLICommand
from manager.constants import logger
from manager.cli.color import Color


class Info(CLICommand):
    doc = """Get detailed info about an instance.
    If multiple instances are matched,
    only the first one will be shown."""
    args_required = ["query"]

    @staticmethod
    def run(cli):
        instances = cli.get_instances_for_action()
        if len(instances) == 0:
            return

        instance = instances[0]
        flags = instance.get_flags()

        status_text = "stopped"
        status_color = Color.RED
        if instance.is_running():
            status_text = "running"
            status_color = Color.GREEN
        status = Color.apply(status_color, status_text)

        lines = []
        lines.extend(
            [
                f"id: {instance.get_id()}:",
                f"user: {instance.get_user()}",
                f"symbol: {instance.get_symbol()}",
                f"status: {status}",
                f"pid: {instance.get_pid_str()}",
                f"config: {instance.get_config()}",
            ]
        )

        if len(flags) > 0:
            lines.append("\nFlags:")
            lines.extend([f"  {flags[i]}: { flags[i + 1]}"
                          for i in range(0, len(flags), 2)])

        for line in lines:
            logger.info(line)
