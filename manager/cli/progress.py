from threading import Thread, Event
from itertools import cycle
from time import sleep
from sys import stdout


class Progress(Thread):
    def __init__(self, event: Event, initial_message: str = "") -> None:
        super().__init__()

        self.states = ["⠷", "⠯", "⠟", "⠻", "⠽", "⠾"]
        self.message = ""
        self.finished = event

        if initial_message is not None:
            self.update(initial_message)

    def run(self):
        for char in cycle(self.states):
            if self.finished.is_set():
                break

            stdout.write(f"\r{char} {self.message}")
            stdout.flush()
            sleep(0.2)

    def update(self, message: str):
        if not self.is_alive():
            self.start()

        self.message = message

    def finish(self, message: str = "Finished"):
        self.finished.set()
        stdout.write(f"\r{message}\n")
        stdout.flush()
