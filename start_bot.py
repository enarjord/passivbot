import sys
from subprocess import Popen
from time import sleep


max_n_restarts = 30

restart_k = 0

while True:
    try:
        print(f"\nStarting {sys.argv[1]} {sys.argv[2]} {sys.argv[3]}")
        p = Popen(f"{sys.executable} passivbot.py {sys.argv[1]} {sys.argv[2]} {sys.argv[3]} --nojit", shell=True)
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
