import os
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_yaml_config(path_to_config: str):
    path = os.getenv(path_to_config)

    if path is None:
        return None

    with open(Path(ROOT_DIR / os.getenv(path_to_config)), "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
