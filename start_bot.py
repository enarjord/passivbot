from subprocess import Popen
from time import sleep
import sys

exchange = sys.argv[1]
user = sys.argv[2]

max_n_restarts = 30

restart_k = 0


while True:
    try:
        print(f"\nStarting {exchange} {user}")
        p = Popen(f"python3 {exchange}.py {user}", shell=True)
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
