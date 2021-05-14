import sys
import argparse
from subprocess import Popen
from time import sleep
from passivbot import get_passivbot_parser

args = get_passivbot_parser().parse_args()

max_n_restarts = 30

restart_k = 0

while True:
    try:
        print(f"\nStarting {args.account_name} {args.symbol} {args.live_config_path}")
        p = Popen(f"{sys.executable} passivbot.py {args.account_name} {args.symbol} {args.live_config_path} ", shell=True)
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
