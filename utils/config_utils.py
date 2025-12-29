from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_yaml_config(path_to_config: str):
    if path_to_config is None:
        return None

    with open(path_to_config, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
