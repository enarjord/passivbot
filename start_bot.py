import argparse
import sys
from subprocess import Popen
from time import sleep

argparser = argparse.ArgumentParser(prog="Bot starter", add_help=True, description="Starts the bot.")
argparser.add_argument("-a", "--account_name", type=str, required=True, dest="a",
                       help="The account name to use in the bot. This needs to be specified in api-keys.json")
argparser.add_argument("-s", "--symbol", type=str, required=True, dest="s",
                       help="The symbol to use in the bot. For example, XMRUSDT.")
argparser.add_argument("-c", "--config_path", type=str, required=True, dest="c",
                       help="The path to the config that the bot should use. For example, live_configs/default.json.")
args = argparser.parse_args()
user = args.a
symbol = args.s
path_to_config = args.c

max_n_restarts = 30

restart_k = 0

while True:
    try:
        print(f"\nStarting {user} {symbol} {path_to_config}")
        p = Popen(f"{sys.executable} passivbot.py {user} {symbol} {path_to_config} --nojit", shell=True)
        p.wait()
        restart_k += 1
        if restart_k > max_n_restarts:
            print('max n restarts reached, aborting')
            break
        for k in range(30 - 1, -1, -1):
            print(f"\rbot stopped, attempting restart in {k} seconds", end='   ')
            sleep(1)
    except KeyboardInterrupt:
        break
