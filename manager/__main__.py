from ast import arg
from sys import argv
from cli import CLI
from pm import ProcessManager

if __name__ == '__main__':
    CLI().run_command(argv[1:])
