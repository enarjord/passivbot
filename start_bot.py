from subprocess import Popen
from time import sleep
import sys

user = sys.argv[1]
symbol = sys.argv[2]
path_to_config = sys.argv[3]



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
