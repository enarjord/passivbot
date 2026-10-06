"""CPU-only durable search state, without GPU handles or borrowed array names."""

import os
import pickle
import logging
from copy import deepcopy

from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract


CHECKPOINT_VERSION = 1


def checkpoint_config(config, contract):
    # Snapshot the actual search policy. Re-running config normalization here
    # can fill omitted bounds and change a metadata-only caller's search space.
    snapshot = {section: deepcopy(config[section]) for section in ("backtest", "optimize", "bot")}
    snapshot[CONTRACT_KEY] = contract
    return snapshot


def validate_checkpoint(state, config):
    if (not isinstance(state, dict) or state.get("backend") != "gpu_native"
            or state.get("version") != CHECKPOINT_VERSION):
        raise ValueError("GPU native resume requires a compatible native checkpoint")
    if state.get(CONTRACT_KEY) != build_evaluation_contract(config):
        raise ValueError("GPU native checkpoint evaluation contract changed; start a fresh run")
    from optimize import _resume_config_mismatches

    snapshot = state.get("resume_config")
    if not isinstance(snapshot, dict):
        raise ValueError("GPU native checkpoint is missing its critical run configuration")
    mismatches = _resume_config_mismatches(snapshot, config)
    if mismatches:
        raise ValueError("GPU native checkpoint critical run configuration changed:\n" + "\n".join(mismatches))
    if state.get("phase") not in {"seeds", "generation", "idle"}:
        raise ValueError("GPU native checkpoint has an invalid search phase")
    if state.get("algorithm") is None:
        raise ValueError("GPU native checkpoint is missing its search algorithm")
    if state.get("phase") != "idle" and state.get("population") is None:
        raise ValueError("GPU native checkpoint is missing pending candidates")
    for key in ("sequence", "completed"):
        if isinstance(state.get(key), bool) or not isinstance(state.get(key), int) or state[key] < 0:
            raise ValueError(f"GPU native checkpoint has an invalid {key}")
    return state


def load_checkpoint(path, config):
    if path is None:
        raise ValueError("GPU native resume requires a checkpoint path")
    with open(path, "rb") as source:
        return validate_checkpoint(pickle.load(source), config)


def save_checkpoint(path, state):
    if path is None:
        return
    temporary = str(path) + ".tmp"
    try:
        with open(temporary, "wb") as target:
            pickle.dump(state, target, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        except OSError:
            logging.exception("GPU native checkpoint temporary cleanup failed after write failure")
        raise
