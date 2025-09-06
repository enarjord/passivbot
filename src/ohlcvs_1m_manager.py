import logging
import traceback


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)

ONE_MIN_MS = 60_000


def get_function_name():
    return inspect.currentframe().f_back.f_code.co_name


class OHLCVManager:

    def __init__(self):
        pass

    def create_lock_file(self, filepath):
        try:
            open(f"{filepath}.lock", "w").close()
            return True
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            traceback.print_exc()
            return False
