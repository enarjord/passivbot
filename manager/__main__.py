from sys import argv
from cli import CLI

if __name__ == '__main__':
    CLI().run_command(argv[1:])
