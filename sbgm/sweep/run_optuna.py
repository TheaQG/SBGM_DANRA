"""

"""

import yaml
import optuna
import pathlib
import logging

DEFAULT_CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "default_config.yaml"
DEFAULT_CFG = yaml.safe_load(DEFAULT_CFG_PATH)
SWEEP_SPACE_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "sweep_spaces" / "sbgm_baseline.yaml"
SWEEP_SPACE = yaml.safe_load(SWEEP_SPACE_PATH)

def suggest_from_space(trial, space):
    """
    Suggest parameters from the given space.
    """
    suggested_params = {}
    for key, value in space.items():
        t = spec["_type"]
        if isinstance(value, dict):
            suggested_params[key] = suggest_from_space(trial, value)
        else:
            suggested_params[key] = trial.suggest_categorical(key, value)
    return suggested_params