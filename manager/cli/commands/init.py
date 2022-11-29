from manager.constants import logger, MANAGER_PATH
from manager.cli.cli import CLICommand
from shutil import copyfile
from os import path


class Init(CLICommand):
    doc = """Create a config file.
    Provide a filename argument to create
    a config file with non-default name"""
    args_optional = ["filename"]

    @staticmethod
    def run(cli):
        filename = "config.yaml"
        if len(cli.args) > 0:
            filename = f"{cli.args[0]}.yaml"

        source = path.join(MANAGER_PATH, "config.example.yaml")
        target = path.join(MANAGER_PATH, filename)

        if path.exists(target):
            logger.info(f"Config file already exists -> {target}")
            return

        copyfile(source, target)
        logger.info(f"Created a config file -> {target}")
