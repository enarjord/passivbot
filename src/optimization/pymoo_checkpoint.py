"""Versioned CPU search state bound to its evaluation and search inputs."""

from copy import deepcopy
import logging
import os
import pickle

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3

from optimization.evaluation_contract import CONTRACT_KEY, recorded_evaluation_contract


CHECKPOINT_VERSION = 1


def checkpoint_state(algorithm, config):
    contract = recorded_evaluation_contract(config)
    snapshot = {
        section: deepcopy(config.get(section, {}))
        for section in ("backtest", "optimize", "bot")
    }
    snapshot[CONTRACT_KEY] = contract
    return {
        "backend": "pymoo",
        "version": CHECKPOINT_VERSION,
        "algorithm": algorithm,
        CONTRACT_KEY: contract,
        "resume_config": snapshot,
    }


def load_checkpoint(path, config):
    if path is None:
        raise ValueError("pymoo resume requires a checkpoint path")
    try:
        with open(path, "rb") as source:
            state = pickle.load(source)
    except (ModuleNotFoundError, AttributeError, ImportError) as exc:
        raise ValueError(
            "pymoo checkpoint classes are incompatible; start a fresh run with saved configs as seeds"
        ) from exc
    if (
        not isinstance(state, dict)
        or state.get("backend") != "pymoo"
        or state.get("version") != CHECKPOINT_VERSION
    ):
        raise ValueError(
            "pymoo resume requires a compatible versioned checkpoint; "
            "start a fresh run with saved configs as seeds"
        )
    if not isinstance(state.get("algorithm"), (NSGA2, NSGA3)):
        raise ValueError("pymoo checkpoint is missing its supported NSGA-II or NSGA-III algorithm")
    if state.get(CONTRACT_KEY) != recorded_evaluation_contract(config):
        raise ValueError("pymoo checkpoint evaluation contract changed; start a fresh run")
    snapshot = state.get("resume_config")
    if not isinstance(snapshot, dict) or snapshot.get(CONTRACT_KEY) != state[CONTRACT_KEY]:
        raise ValueError("pymoo checkpoint is missing its critical run configuration")
    from optimize import _resume_config_mismatches

    mismatches = _resume_config_mismatches(snapshot, config)
    if mismatches:
        raise ValueError(
            "pymoo checkpoint critical run configuration changed:\n" + "\n".join(mismatches)
        )
    return state["algorithm"]


def save_checkpoint(path, algorithm, config):
    temporary = str(path) + ".tmp"
    try:
        with open(temporary, "wb") as target:
            pickle.dump(checkpoint_state(algorithm, config), target, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        except OSError:
            logging.exception("pymoo checkpoint temporary cleanup failed after write failure")
        raise
