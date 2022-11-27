from manager.cli.cli import CLICommand
from manager.constants import logger


class Help(CLICommand):
    doc = "Show help"
    args_optional = ["command"]

    @staticmethod
    def run(cli):
        if len(cli.commands_available) == 0:
            logger.info("There are no avilable commands")
            return

        if len(cli.args) > 0:
            name = cli.commands_available.get(cli.args[0])
            if name is None:
                logger.info(
                    "There is no available command: {}".format(cli.args[0]))
                return

            logger.info("Help for {}:".format(cli.args[0]))
            for line in Help.get_docs():
                logger.info(line)
            return

        logger.info("Available commands:")

        lines = [
            "Usage: manager <command> [args]",
            "\n  CLI for managing instances of PassivBot\n",
            "Commands:",
        ]

        for name, command in cli.commands_available.items():
            doc_lines = command.get_docs()

            lines.append("  {:<8} - {}".format(name, doc_lines[0]))
            for line in doc_lines[1:]:
                lines.append("  {:<10} {}".format("", line))

        lines.append("\nFlags:")
        for flag in cli.flags_available.values():
            variants = ", ".join(flag["variants"])
            lines.append(
                "  {:<15} - {}".format(
                    variants,
                    flag.get("doc"),
                )
            )

        for line in lines:
            logger.info(line)
